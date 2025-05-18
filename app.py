# enhanced_app.py
import streamlit as st
from PIL import Image
import time
import numpy as np
import io
from app.htr_engine import EnhancedHTREngine
from app.image_preprocessing import preprocess_image

# Set page configuration
st.set_page_config(
    page_title="Document Handwritten Text Recognition",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'recognition_results' not in st.session_state:
    st.session_state.recognition_results = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'segmentation_visualization' not in st.session_state:
    st.session_state.segmentation_visualization = None

# Initialize the Enhanced HTR engine with empty line threshold
@st.cache_resource
def load_enhanced_htr_engine(empty_line_threshold=0.01):
    return EnhancedHTREngine(empty_line_threshold=empty_line_threshold)

# Title and description
st.title("📄 Document Handwritten Text Recognition")
st.markdown("""
This application recognizes handwritten text from complete documents, paragraphs, or pages.
The system automatically segments the document into lines or paragraphs and processes each segment individually.
""")

# Sidebar options
st.sidebar.header("⚙️ Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["microsoft/trocr-base-handwritten", "microsoft/trocr-large-handwritten"],
    help="Choose the TrOCR model for text recognition"
)

processing_mode = st.sidebar.selectbox(
    "Processing Mode",
    ["lines", "paragraphs"],
    help="Lines: Process each line separately\nParagraphs: Group lines into paragraphs"
)

# Preprocessing options
st.sidebar.subheader("🔧 Preprocessing Options")
denoise = st.sidebar.checkbox("Apply Denoising", value=True, help="Remove noise from the image")
contrast_enhance = st.sidebar.checkbox("Enhance Contrast", value=True, help="Improve image contrast")
deskew = st.sidebar.checkbox("Deskew Image", value=False, help="Straighten skewed text")

# Segmentation options
st.sidebar.subheader("✂️ Segmentation Options")
if processing_mode == "lines":
    line_padding = st.sidebar.slider("Line Padding", 0, 20, 5, help="Padding around each line segment")
else:
    min_paragraph_gap = st.sidebar.slider("Min Paragraph Gap", 10, 50, 30, help="Minimum gap to separate paragraphs")

# Empty line detection options
st.sidebar.subheader("🔍 Empty Line Detection")
empty_line_threshold = st.sidebar.slider(
    "Empty Line Threshold", 
    0.001, 0.05, 0.01, 
    step=0.001,
    format="%.3f",
    help="Threshold for detecting empty lines (proportion of pixels that must be text)"
)
show_empty_lines = st.sidebar.checkbox("Show Empty Lines", value=False, help="Include empty lines in results")

# Visualization options
show_segmentation = st.sidebar.checkbox("Show Segmentation", value=False, help="Display segmentation overlay")

# Load the HTR engine with configured threshold
htr_engine = load_enhanced_htr_engine(empty_line_threshold)

# Function to process and recognize document
def process_and_recognize_document(image):
    """Process document and recognize text."""
    try:
        with st.spinner("Processing document..."):
            # Apply preprocessing
            processed_image = preprocess_image(
                image,
                denoise=denoise,
                contrast_enhance=contrast_enhance,
                deskew=deskew
            )
            st.session_state.processed_image = processed_image
            
            # Create segmentation visualization if requested
            if show_segmentation:
                vis_image = htr_engine.visualize_segmentation(processed_image, processing_mode)
                st.session_state.segmentation_visualization = vis_image
            
            # Perform document-level HTR
            start_time = time.time()
            results = htr_engine.recognize_document(
                processed_image,
                model_name=model_option,
                processing_mode=processing_mode
            )
            end_time = time.time()
            
            # Add processing time to results
            results['processing_time'] = end_time - start_time
            results['model_used'] = model_option
            results['preprocessing_settings'] = {
                'denoise': denoise,
                'contrast_enhance': contrast_enhance,
                'deskew': deskew,
                'empty_line_threshold': empty_line_threshold,
                'show_empty_lines': show_empty_lines
            }
            
            # Filter out empty lines if requested
            if not show_empty_lines and processing_mode == "lines":
                results['lines'] = [line for line in results['lines'] if line['text'].strip()]
                results['full_text'] = "\n".join([line['text'] for line in results['lines'] if line['text'].strip()])
            
            # Store results in session state
            st.session_state.recognition_results = results
            
            return True
    except Exception as e:
        st.error(f"Error during recognition: {str(e)}")
        return False

# Main content - Image Upload Section
uploaded_file = st.file_uploader(
    "Upload a document image with handwritten text", 
    type=["png", "jpg", "jpeg"],
    help="Upload an image containing handwritten text (pages, paragraphs, or multiple lines)"
)

if uploaded_file is not None:
    # Load and display the original image
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Create layout columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)
        
        # Process button
        if st.button("🔍 Recognize Document Text", type="primary", use_container_width=True):
            success = process_and_recognize_document(image)
    
    # Display processed image and segmentation
    with col2:
        if st.session_state.processed_image is not None:
            st.subheader("🔧 Processed Image")
            st.image(st.session_state.processed_image, use_column_width=True)
            
        if st.session_state.segmentation_visualization is not None:
            st.subheader("✂️ Segmentation Visualization")
            st.image(st.session_state.segmentation_visualization, use_column_width=True)
    
    # Display results if available
    if st.session_state.recognition_results is not None:
        results = st.session_state.recognition_results
        
        st.markdown("---")
        st.subheader("📝 Recognition Results")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Full Text", "Detailed Results", "Statistics"])
        
        with tab1:
            # Display full recognized text
            if results['full_text'].strip():
                st.success("**Complete Recognized Text:**")
                st.text_area(
                    "Recognized Text",
                    value=results['full_text'],
                    height=300,
                    help="Full text extracted from the document"
                )
                
                # Download button for full text
                st.download_button(
                    label="📝 Download Full Text",
                    data=results['full_text'],
                    file_name="recognized_document.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.warning("No text was recognized. Try adjusting preprocessing settings or using a clearer image.")
        
        with tab2:
            # Display detailed line/paragraph results
            if results['processing_mode'] == 'lines' and results['lines']:
                st.subheader("📏 Line-by-Line Results")
                
                for i, line in enumerate(results['lines']):
                    if line['text'].strip() or results['preprocessing_settings'].get('show_empty_lines', False):
                        confidence_color = "red" if line['confidence'] < 0.5 else "green"
                        line_label = f"Line {i+1}" if line['text'].strip() else f"Empty Line {i+1}"
                        with st.expander(f"{line_label} (Confidence: {line['confidence']:.3f})"):
                            if line['text'].strip():
                                st.write(f"**Text:** {line['text']}")
                            else:
                                st.write("**Text:** <empty line>")
                            st.write(f"**Confidence:** :{confidence_color}[{line['confidence']:.3f}]")
                            st.write(f"**Bounding Box:** {line['bbox']}")
                            
            elif results['processing_mode'] == 'paragraphs' and results['paragraphs']:
                st.subheader("📄 Paragraph Results")
                
                for i, para in enumerate(results['paragraphs']):
                    with st.expander(f"Paragraph {i+1} (Lines: {para['line_count']}, Confidence: {para['confidence']:.3f})"):
                        st.write(f"**Text:** {para['text']}")
                        st.write(f"**Confidence:** {para['confidence']:.3f}")
                        st.write(f"**Number of Lines:** {para['line_count']}")
        
        with tab3:
            # Display statistics
            st.subheader("📊 Processing Statistics")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Overall Confidence", f"{results['overall_confidence']:.3f}")
                st.metric("Processing Time", f"{results['processing_time']:.2f}s")
                
            with col_stat2:
                if results['processing_mode'] == 'lines':
                    st.metric("Total Lines", len(results['lines']))
                    non_empty_lines = len([l for l in results['lines'] if l['text'].strip()])
                    st.metric("Lines with Text", non_empty_lines)
                    if not results['preprocessing_settings'].get('show_empty_lines', False):
                        st.metric("Empty Lines Filtered", len(results['lines']) - non_empty_lines)
                else:
                    st.metric("Total Paragraphs", len(results['paragraphs']))
                    total_lines = sum(p['line_count'] for p in results['paragraphs'])
                    st.metric("Total Lines", total_lines)
                    
            with col_stat3:
                st.metric("Model Used", results['model_used'].split('/')[-1])
                word_count = len(results['full_text'].split()) if results['full_text'] else 0
                st.metric("Word Count", word_count)
            
            # Preprocessing settings
            st.subheader("🔧 Preprocessing Settings")
            settings = results.get('preprocessing_settings', {})
            st.json(settings)

# Clear results button in sidebar
if st.session_state.recognition_results is not None:
    if st.sidebar.button("🗑️ Clear Results"):
        st.session_state.recognition_results = None
        st.session_state.processed_image = None
        st.session_state.segmentation_visualization = None
        st.rerun()

# Example usage section
with st.expander("📋 Example Use Cases"):
    st.markdown("""
    **This enhanced system is perfect for:**
    
    1. **Complete Documents**: Essays, reports, letters
    2. **Multiple Paragraphs**: Structured text with clear spacing
    3. **Mixed Content**: Documents with headers, paragraphs, and lists
    4. **Research Notes**: Handwritten research papers or notes
    5. **Forms**: Filled-out forms with multiple text fields
    6. **Historical Documents**: Old manuscripts or handwritten records
    
    **Best Practices:**
    
    - Use 'lines' mode for simple line-by-line text
    - Use 'paragraphs' mode for structured documents
    - Enable segmentation visualization to verify proper text detection
    - Adjust padding/gap settings based on your document layout
    - Try different preprocessing combinations for optimal results
    - Adjust the empty line threshold to properly filter out spaces between text
    """)