import streamlit as st
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Azure Imports ---
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================
st.set_page_config(page_title="AI Book Summarizer", page_icon="📚", layout="wide")

if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "current_book" not in st.session_state:
    st.session_state.current_book = None

@dataclass
class BookMetadata:
    title: str
    author: str
    filename: str
    publisher: str = ""
    published_date: str = ""
    description: str = ""

@dataclass
class BookInfo:
    filename: str
    title: str
    author: str

# ============================================================================
# SIDEBAR: CREDENTIALS
# ============================================================================
st.sidebar.header("⚙️ Azure Configuration")

AZURE_STORAGE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-12-01-preview")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME")

creds_present = all([AZURE_STORAGE_CONN_STR, CONTAINER_NAME, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX])

with st.sidebar:
    st.header("⚙️ System Status")
    if creds_present:
        st.success("✅ Credentials Loaded from .env")
    else:
        st.error("❌ Missing Credentials in .env")
        st.info("Please check your .env file contains all required Azure keys.")

# ============================================================================
# LOGIC: BOOK DISCOVERY & PARSING
# ============================================================================
def parse_book_filename(filename: str) -> Tuple[str, str]:
    name_without_ext = filename.replace('.pdf', '').strip()
    common_suffixes = [' - libgen.li', ' - z-lib.org', ' - z-library', ' - libgen', ' - zlibrary']
    for suffix in common_suffixes:
        if name_without_ext.lower().endswith(suffix.lower()):
            name_without_ext = name_without_ext[:-len(suffix)].strip()
            break
    
    if ' - ' in name_without_ext:
        parts = name_without_ext.split(' - ')
        if len(parts) >= 2:
            return parts[1].strip(), parts[0].strip() 
        else:
            return parts[0].strip(), "Unknown Author"
    elif '(' in name_without_ext and ')' in name_without_ext:
        title = name_without_ext.split('(')[0].strip()
        author = name_without_ext.split('(')[1].replace(')', '').strip()
        return title, author
    else:
        return name_without_ext.strip(), "Unknown Author"

@st.cache_data(show_spinner=False)
def get_all_books_info(_conn_str, _container) -> List[BookInfo]:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(_conn_str)
        container_client = blob_service_client.get_container_client(_container)
        books_info = []
        for blob in container_client.list_blobs():
            if blob.name.endswith('.pdf'):
                title, author = parse_book_filename(blob.name)
                books_info.append(BookInfo(filename=blob.name, title=title, author=author))
        return books_info
    except Exception as e:
        st.error(f"Error connecting to storage: {e}")
        return []

def search_book_by_criteria(books_info: List[BookInfo], title: str = None, author: str = None) -> List[BookInfo]:
    """Filter books based on title or author."""
    matches = []
    for book in books_info:
        title_match = True
        author_match = True
        if title:
            title_match = title.lower() in book.title.lower()
        if author:
            author_match = author.lower() in book.author.lower()
        if title_match and author_match:
            matches.append(book)
    return matches

# ============================================================================
# PROMPT GENERATION HELPER
# ============================================================================
def get_default_prompt_template(language: str) -> str:
    """Returns the default prompt string with the correct language callouts pre-filled."""
    
    callout_translations = {
        "English": {"did_you_know": "Did you know?", "try_this": "Try this:", "heres_the_thing": "Here's the thing:", "remember": "Remember:", "lets_be_honest": "Let's be honest:", "the_bottom_line": "The bottom line:", "sound_familiar": "Sound familiar?", "key_takeaway": "Key takeaway:", "think_about_it": "Think about it:", "the_truth_is": "The truth is:"},
        "Spanish": {"did_you_know": "¿Sabías que?", "try_this": "Prueba esto:", "heres_the_thing": "Esto es lo importante:", "remember": "Recuerda:", "lets_be_honest": "Seamos honestos:", "the_bottom_line": "En resumen:", "sound_familiar": "¿Te suena familiar?", "key_takeaway": "Punto clave:", "think_about_it": "Piénsalo:", "the_truth_is": "La verdad es:"},
        "Turkish": {"did_you_know": "Biliyor muydunuz?", "try_this": "Bunu deneyin:", "heres_the_thing": "Şöyle bir şey var:", "remember": "Unutmayın:", "lets_be_honest": "Açıkçası:", "the_bottom_line": "Özetle:", "sound_familiar": "Tanıdık geliyor mu?", "key_takeaway": "Önemli nokta:", "think_about_it": "Bir düşünün:", "the_truth_is": "Gerçek şu ki:"},
        "French": {"did_you_know": "Le saviez-vous?", "try_this": "Essayez ceci:", "heres_the_thing": "Voici la chose:", "remember": "Rappelez-vous:", "lets_be_honest": "Soyons honnêtes:", "the_bottom_line": "L'essentiel:", "sound_familiar": "Ça vous dit quelque chose?", "key_takeaway": "Point clé:", "think_about_it": "Pensez-y:", "the_truth_is": "La vérité est:"},
        "German": {"did_you_know": "Wussten Sie?", "try_this": "Versuchen Sie dies:", "heres_the_thing": "Die Sache ist:", "remember": "Denken Sie daran:", "lets_be_honest": "Seien wir ehrlich:", "the_bottom_line": "Unterm Strich:", "sound_familiar": "Kommt Ihnen bekannt vor?", "key_takeaway": "Wichtigster Punkt:", "think_about_it": "Denken Sie darüber nach:", "the_truth_is": "Die Wahrheit ist:"}
    }
    
    # Default to English if language not found
    c = callout_translations.get(language, callout_translations["English"])
    
    # FIX: Replaced {self.summary_language} with {language} below
    return f"""
You are creating a COMPREHENSIVE, LONG, engaging, conversational book summary for a Personal Growth & Self-Help book.
This should be an EXTENSIVE and DETAILED summary that thoroughly covers all major concepts from the book.
Your goal is to make complex ideas accessible and keep readers engaged throughout a 12-15 minute read.

**CRITICAL LANGUAGE REQUIREMENT:**
- Write the ENTIRE summary in {language}
- ALL text including headers, callouts, phrases, and content MUST be in {language}
- Use these specific callout titles if necessary in {language}:
  * "{c['did_you_know']}" (for fascinating facts)
  * "{c['try_this']}" (for actionable steps)
  * "{c['heres_the_thing']}" (for important insights)
  * "{c['remember']}" (for key principles)
  * "{c['lets_be_honest']}" (for relatable truths)
  * "{c['the_bottom_line']}" (for summarizing takeaways)
  * "{c['sound_familiar']}" (for connection with readers)
  * "{c['key_takeaway']}" (for main points)
  * "{c['think_about_it']}" (for reflection prompts)
  * "{c['the_truth_is']}" (for powerful statements)

**GENRE FOCUS: Personal Growth & Self-Help**
This book belongs to the Personal Growth & Self-Help genre. Your summary should:
- Focus on practical self-improvement strategies and actionable advice
- Address common personal challenges (mindset, habits, relationships, success, fulfillment)
- Emphasize personal transformation and growth
- Include psychological insights and behavioral change principles
- Connect concepts to the reader's personal journey
- Balance inspiration with practical, implementable steps
- Address both internal (mindset, beliefs) and external (actions, habits) changes
- Use relatable life scenarios and everyday examples

**LENGTH REQUIREMENTS - CRITICAL:**
1. The summary MUST be EXTENSIVE and COMPREHENSIVE - aim for 10,000-15,000 words minimum
2. This is a 12-15 minute read - it needs to be LONG and provide substantial value
3. Cover 8-12 major concepts/principles from the book
4. Each major concept should have 800-1,500 words of detailed explanation
5. Include multiple examples, scenarios, and practical applications for each concept
6. DO NOT create a brief summary - expand every concept thoroughly
7. Each major section should have 5-8 paragraphs of detailed explanation
8. Don't rush through concepts - take time to explain them thoroughly with examples
9. The reader should feel they've gotten immense value without reading the full book

**STRUCTURE - COMPREHENSIVE Coverage:**

**Opening (200-300 words):**
- Start with an engaging hook - a provocative question or relatable scenario about personal growth
- Set up the main problem or challenge in personal development that the book addresses
- Create curiosity about the transformation or insights to come
- Make it personally relevant to the reader's life journey

**8-12 MAJOR SECTIONS (each 800-1,500 words):**

Each major section MUST include ALL of these elements:
- A bold, descriptive, engaging header in {language} (make it a question or provocative statement)
- Opening hook: relatable personal growth question or scenario (1 paragraph)
- Core concept explanation: explain the self-help principle clearly (2-3 paragraphs)
- Problem identification: what personal challenge does this address? (1-2 paragraphs)
- Detailed explanation: dive deep into HOW and WHY this principle works psychologically (2-3 paragraphs)
- Multiple real-world examples: show the principle in everyday life situations (2-3 paragraphs)
- Contrasting perspectives: common mistakes vs. effective approaches (1-2 paragraphs)
- At least one "{c['did_you_know']}" callout with a surprising psychological fact or research finding
- Practical implications: how this applies to reader's personal growth journey (1-2 paragraphs)
- A "{c['try_this']}" section with 2-3 specific, actionable steps readers can implement today
- Smooth transition to next concept (1 paragraph)

**Conclusion (400-600 words):**
- Synthesize key insights from all sections into a cohesive personal growth framework
- Provide an empowering final message about the reader's transformation potential
- Include comprehensive "{c['try_this']}" section with 8-12 concrete action items for immediate implementation
- End with motivational encouragement specific to personal growth and self-improvement
- Address the reader's ability to create lasting change

Extract 3-6 cards from EACH section you wrote above after Conclusion. Format each card exactly like this:
**CARDS:*
***CARD TYPE: Quote***
SECTION: [Which section this came from]
TITLE: [Short descriptive title]
CONTENT: [The actual quote or key content, 50-150 words]
TAGS: [tag1, tag2, tag3]

***CARD TYPE: Insight***
SECTION: [Which section this came from]
TITLE: [Short descriptive title]
CONTENT: [Key insight or lesson, 100-200 words]
TAGS: [tag1, tag2, tag3]

***CARD TYPE: Fact***
SECTION: [Which section this came from]
TITLE: [Short descriptive title]
CONTENT: [Research finding or statistic, 80-150 words]
TAGS: [tag1, tag2, tag3]

***CARD TYPE: Concept***
SECTION: [Which section this came from]
TITLE: [Short descriptive title]
CONTENT: [Model or framework explanation, 100-200 words]
TAGS: [tag1, tag2, tag3]


**CONVERSATIONAL TONE - Personal Growth Style:**
- Write as if you're a trusted mentor speaking to someone on their growth journey
- Use "you" and "we" to create personal connection (in {language})
- Keep language accessible but don't sacrifice depth or psychological insights
- Use short, punchy statements for emphasis on key mindset shifts
- Include rhetorical questions that prompt self-reflection
- Share the principles like personal wisdom learned through experience
- Be encouraging yet realistic about the work required for change
- Acknowledge struggles while maintaining optimism about growth potential

**CONTENT FEATURES for Personal Growth:**
- Problem-solution format showing path from struggle to transformation
- Real-world scenarios from work, relationships, daily habits, and life decisions
- Personal reflection questions to help readers apply concepts
- Psychological principles explained in accessible terms
- Mindset shifts and reframes of common limiting beliefs
- Habit formation and behavior change strategies
- Stories of transformation and before/after contrasts
- Balance between inner work (mindset) and outer action (habits)
- Address resistance to change and how to overcome it

**WRITING STYLE - DETAILED but ENGAGING:**
- Each self-help principle deserves thorough, multi-paragraph exploration
- Use 5-8 paragraphs per major section
- Tell mini-stories or scenarios that readers can see themselves in
- Use analogies and metaphors from everyday life
- Break up longer explanations with questions, callouts, or examples
- Vary sentence length for rhythm and emotional impact
- Balance motivation with practical how-to guidance
- Never be brief when you can provide thorough, actionable insights

**FORMATTING RULES:**
- Use bold for section headers and key concepts in {language}
- Keep individual paragraphs short (3-5 sentences max) but use MANY paragraphs
- Use bullet points for frameworks, habit lists, or action steps
- Break up long explanations every few paragraphs with callouts or examples
- NO generic headers - make them personal, provocative, and in {language}

**CRITICAL - Make it COMPREHENSIVE:**
- DO NOT create a brief overview - this needs depth and length
- Each principle deserves 800-1,500 words with multiple examples from personal growth contexts
- Include scenarios from relationships, career, habits, mindset, and life decisions
- The reader should feel equipped to actually transform their life with these insights
- Aim for 10,000-15,000 words total
- Quality AND quantity - be thorough, practical, and deeply engaging

Book Title: {{title}}
Author: {{author}}
Genre: Personal Growth & Self-Help

Content:
{{content}}

Now create an EXTENSIVE, LONG, engaging, conversational summary (10,000-15,000 words) IN {language} that thoroughly covers all major personal growth concepts from the book. Remember: ALL text including headers, callouts, and phrases must be in {language}. Make readers feel like they're having a transformative conversation with a wise mentor who's taking the time to explain each principle in detail with plenty of real-life examples and actionable steps for personal growth.

Book Title: {{title}}
Author: {{author}}
Genre: Personal Growth & Self-Help

Content:
{{content}}

Now create your EXTENSIVE, LONG, engaging summary (10,000-15,000 words).
"""

# ============================================================================
# LOGIC: SUMMARIZER CLASS
# ============================================================================
class HeadwayStyleBookSummarizer:
    def __init__(self, summary_language: str):
        self.summary_language = summary_language
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self.openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return response.data[0].embedding
        except Exception as e:
            st.error(f"❌ Error generating embedding: {e}")
            return []
    
    def generate_retrieval_queries(self, book_metadata: BookMetadata) -> List[str]:
        t = book_metadata.title
        a = book_metadata.author
        queries = [f"{t}", f"{t} summary", f"{t} key concepts", f"{t} main ideas", f"{a} {t}",
                   f"key takeaways {t}", f"actionable advice {t}", f"lessons from {t}"]
        return list(set(queries))
    
    def retrieve_documents(self, queries: List[str], max_docs_per_query: int = 100) -> Dict[str, List[str]]:
        all_documents = {}
        progress_text = "Retrieving documents from Azure Search..."
        my_bar = st.progress(0, text=progress_text)
        total = len(queries)
        
        for i, query in enumerate(queries):
            try:
                my_bar.progress(int((i / total) * 100), text=f"Searching: {query}")
                emb = self.generate_embedding(query)
                if not emb: continue
                
                sc = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
                vq = VectorizedQuery(vector=emb, k_nearest_neighbors=50, fields="text_vector")
                results = sc.search(search_text='*', vector_queries=[vq], select=["chunk"], top=max_docs_per_query)
                
                docs = [r["chunk"] for r in results if "chunk" in r]
                all_documents[query] = docs
                time.sleep(0.2)
            except Exception as e:
                all_documents[query] = []
        my_bar.empty()
        return all_documents
    
    def generate_headway_style_summary(self, metadata: BookMetadata, content: str, prompt_template: str) -> str:
        """
        Generates summary using the user-provided prompt string.
        Replaces {title}, {author}, and {content} placeholders.
        """
        if not content.strip(): return "Summary could not be generated - no content available."
        
        # Safe replacement of placeholders in the prompt template
        # Note: We use .replace() because f-strings are evaluated at runtime, 
        # but the template comes from the UI text box as a raw string.
        final_prompt = prompt_template.replace("{{title}}", metadata.title)
        final_prompt = final_prompt.replace("{{author}}", metadata.author)
        
        # Handle single brackets just in case
        final_prompt = final_prompt.replace("{title}", metadata.title)
        final_prompt = final_prompt.replace("{author}", metadata.author)
        
        # Insert content (Limit to 15k chars to prevent context window overflow)
        final_prompt = final_prompt.replace("{{content}}", content[:15000])
        final_prompt = final_prompt.replace("{content}", content[:15000])
        
        try:
            response = self.openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=4096,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

# ============================================================================
# MAIN UI
# ============================================================================
st.title("📖 Deep Dive Book Summarizer")

if not creds_present:
    st.warning("⚠️ Please enter all Azure Configuration details in the sidebar.")
else:
    with st.spinner("Loading library from Azure Storage..."):
        all_books = get_all_books_info(AZURE_STORAGE_CONN_STR, CONTAINER_NAME)

    if all_books:
        st.markdown("### 🔍 Find a Book")
        
        # --- INTERACTIVE SEARCH LOGIC ---
        col_search1, col_search2 = st.columns([1, 3])
        
        with col_search1:
            search_mode = st.radio(
                "Search by:",
                ["Title", "Author", "Both", "View All"],
                key="search_mode"
            )
        
        with col_search2:
            matches = []
            if search_mode == "Title":
                title_query = st.text_input("Enter book title (partial match):")
                if title_query:
                    matches = search_book_by_criteria(all_books, title=title_query)
            
            elif search_mode == "Author":
                author_query = st.text_input("Enter author name (partial match):")
                if author_query:
                    matches = search_book_by_criteria(all_books, author=author_query)
            
            elif search_mode == "Both":
                c1, c2 = st.columns(2)
                t_q = c1.text_input("Title:")
                a_q = c2.text_input("Author:")
                if t_q or a_q:
                    matches = search_book_by_criteria(all_books, title=t_q, author=a_q)
            
            elif search_mode == "View All":
                matches = all_books

        # --- SELECTION & GENERATION ---
        if matches:
            st.success(f"Found {len(matches)} matching book(s).")
            book_map = {f"{b.title} ({b.author})": b for b in matches}
            
            col_sel1, col_sel2 = st.columns([3, 1])
            with col_sel1:
                selected_key = st.selectbox("Select the book to summarize:", options=list(book_map.keys()))
                selected_book = book_map[selected_key]
            
            with col_sel2:
                language = st.selectbox("Language:", ["English", "Spanish", "French", "Turkish", "German"])

            # --- EDITABLE PROMPT SECTION ---
            st.markdown("### 🛠️ Prompt Configuration")
            with st.expander("📝 Advanced: Edit System Prompt (Click to Open)", expanded=False):
                st.info("The text below is the exact instruction sent to GPT-4. You can edit it to change the tone, length, or structure. Do not remove `{title}`, `{author}`, or `{content}` tags.")
                
                # Get the default template dynamically based on the selected language
                default_template = get_default_prompt_template(language)
                
                # User can edit this
                user_prompt = st.text_area("System Prompt", value=default_template, height=400)

            if st.button("🚀 Generate Summary", type="primary"):
                st.session_state.current_book = selected_book.title
                
                summarizer = HeadwayStyleBookSummarizer(summary_language=language)
                metadata = BookMetadata(title=selected_book.title, author=selected_book.author, filename=selected_book.filename)
                
                with st.status("Processing...", expanded=True) as status:
                    st.write("Generating queries...")
                    queries = summarizer.generate_retrieval_queries(metadata)
                    
                    st.write("Retrieving content...")
                    retrieved_docs = summarizer.retrieve_documents(queries)
                    
                    all_content = ""
                    for q, docs in retrieved_docs.items():
                        if docs: all_content += "\n".join(docs) + "\n"
                    
                    if not all_content.strip():
                        status.update(label="Failed", state="error")
                        st.error("No content found.")
                    else:
                        status.update(label="Generating Summary (this takes ~1 min)...", state="running")
                        
                        # CALL THE METHOD WITH THE USER PROMPT
                        summary_text = summarizer.generate_headway_style_summary(metadata, all_content, user_prompt)
                        
                        st.session_state.summary_result = summary_text
                        status.update(label="Complete!", state="complete", expanded=False)

        elif search_mode != "View All":
            st.info("Enter a search term to find books.")
            
    # --- DISPLAY RESULT ---
    if st.session_state.summary_result:
        st.divider()
        st.subheader(f"Summary: {st.session_state.current_book}")
        st.download_button("📥 Download Markdown", st.session_state.summary_result, file_name="summary.md")
        st.markdown(st.session_state.summary_result)
