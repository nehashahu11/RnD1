import streamlit as st
from huggingface_hub import InferenceClient
import xml.etree.ElementTree as ET
from huggingface_hub.errors import HfHubHTTPError
import time

# Initialize HF client
API_KEY = "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"
client = InferenceClient(provider="hf-inference", api_key=API_KEY)

# Models
VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Retry helper

def retryable_call(func, *args, **kwargs):
    backoff = [1, 2, 4]
    for delay in backoff:
        try:
            return func(*args, **kwargs)
        except HfHubHTTPError as e:
            code = getattr(e, 'status_code', None) or getattr(getattr(e, 'response', None), 'status_code', None)
            if code == 429:
                time.sleep(delay)
                continue
            raise
    return func(*args, **kwargs)

# Extract XML from image

def extract_table_xml_from_image(url: str) -> str:
    system = {"role": "system", "content": (
        "You convert an image of a data table into clean, well-formed XML. Output only XML."
    )}
    user = {"role": "user", "content": [
        {"type": "text", "text": "Extract all rows and columns as <table>,<tr>,<th>,<td>."},
        {"type": "image_url", "image_url": {"url": url}}
    ]}
    resp = retryable_call(
        client.chat.completions.create,
        model=VISION_MODEL,
        messages=[system, user],
        temperature=0.0,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()

# Linearize XML

def linearize_table_xml(xml: str) -> str:
    root = ET.fromstring(xml)
    rows = root.findall('.//tr') or root.findall('.//row')
    lines = []
    for r in rows:
        cells = [c.text.strip() for c in r if c.text and c.text.strip()]
        if cells:
            lines.append('<row>' + '<sep>'.join(cells) + '</row>')
    return '<rows>\n' + '\n'.join(lines) + '\n</rows>'

# Summarization functions (bb/img/xml) omitted for brevity...
# score_subjectivity_features omitted for brevity...

# Main UI

def main():
    st.set_page_config(page_title="Table-to-Text Multi-Sport", layout="wide")
    st.title("Multi-Sport Table Interpretation & Analysis")

    mode = st.selectbox("Select Input Type:", [
        "Basketball XML", "Basketball Image",
        "Football XML",   "Football Image"
    ])

    # Initialize session state
    if 'xml' not in st.session_state:
        st.session_state.xml = ''

    # Input section
    if "XML" in mode:
        st.subheader("Paste XML Table")
        st.session_state.xml = st.text_area("Table XML:", value=st.session_state.xml, height=200)
        if st.session_state.xml:
            st.subheader("Raw XML Preview")
            st.code(st.session_state.xml, language='xml')
    else:
        st.subheader("Enter Table Image URL")
        url = st.text_input("Image URL:")
        if st.button("Extract XML") and url:
            with st.spinner("Extracting XML..."):
                try:
                    st.session_state.xml = extract_table_xml_from_image(url)
                except Exception as e:
                    st.error(f"Error: {e}")
                else:
                    st.subheader("Extracted XML")
                    st.code(st.session_state.xml, language='xml')

    # Once xml exists
    if st.session_state.xml:
        flat = linearize_table_xml(st.session_state.xml)

        # Generate summary
        if st.button("Generate Summary"):
            with st.spinner("Generating Summary..."):
                summary = generate_summary_bb(flat) if "Basketball" in mode else generate_summary_fb(flat)
                st.session_state.summary = summary
                st.subheader("Summary")
                st.write(summary)

        # Rate subjectivity
        if 'summary' in st.session_state and 'xml' in st.session_state:
            if st.button("Rate Subjectivity"):
                with st.spinner("Rating..."):
                    st.session_state.score = score_subjectivity_features(st.session_state.summary)
                    st.subheader("Subjectivity Scores")
                    # display all metrics in one row
                    cols = st.columns(len(st.session_state.score))
                    for idx, (feature, val) in enumerate(st.session_state.score.items()):
                        cols[idx].metric(label=feature, value=val)

        # Final comparison view
        if 'summary' in st.session_state and 'score' in st.session_state:
            st.markdown("---")
            st.subheader("Final Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Extracted XML**")
                st.code(st.session_state.xml, language='xml')
            with col2:
                st.markdown("**Summary**")
                st.write(st.session_state.summary)
            with col3:
                st.markdown("**Subjectivity Scores**")
                for feature, val in st.session_state.score.items():
                    st.write(f"- **{feature}**: {val}")

if __name__ == "__main__":
    main()
