"""
Interactive Chat Interface for RAG Complaint Chatbot.

This Gradio application provides a user-friendly interface for non-technical users
to query customer complaint data using natural language.
"""

import gradio as gr
import logging
from typing import Tuple, List
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None


def initialize_pipeline():
    """Initialize the RAG pipeline on startup."""
    global pipeline

    logger.info("Initializing RAG Pipeline...")

    try:
        pipeline = RAGPipeline(
            vector_store_path="vector_store",
            collection_name="complaints",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            llm_model="HuggingFaceH4/zephyr-7b-beta",
            top_k=5,
            template_name="default",
            device="cpu",  # Change to "cuda" if GPU available
            max_new_tokens=512,
            temperature=0.7,
        )
        logger.info("RAG Pipeline initialized successfully!")
        return "✅ System ready! Ask your first question."

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return f"❌ Error initializing system: {str(e)}"


def format_sources(sources: List[dict]) -> str:
    """
    Format retrieved source chunks for display.

    Args:
        sources: List of source document dictionaries.

    Returns:
        Formatted HTML string of sources.
    """
    if not sources:
        return "<p><i>No sources retrieved.</i></p>"

    html = "<div style='margin-top: 20px;'>"
    html += "<h3 style='color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 5px;'>📄 Retrieved Sources</h3>"

    for i, source in enumerate(sources, 1):
        # Extract metadata
        complaint_id = source.get("complaint_id", "N/A")
        product = source.get("product_category", source.get("product", "N/A"))
        issue = source.get("issue", "N/A")
        text = source.get("text", source.get("document", "N/A"))
        distance = source.get("distance", source.get("score", 0))

        # Calculate similarity score (convert distance to similarity)
        similarity = round((1 - distance) * 100, 1) if distance else "N/A"

        # Create source card
        html += f"""
        <div style='background-color: #f8f9fa; border-left: 4px solid #1f77b4; 
                    padding: 15px; margin: 10px 0; border-radius: 5px;'>
            <div style='font-weight: bold; color: #1f77b4; margin-bottom: 8px;'>
                Source {i} (Similarity: {similarity}%)
            </div>
            <div style='margin-bottom: 5px;'>
                <strong>Product:</strong> {product} | 
                <strong>Issue:</strong> {issue} | 
                <strong>ID:</strong> {complaint_id}
            </div>
            <div style='background-color: white; padding: 10px; border-radius: 3px; 
                        margin-top: 8px; font-style: italic; color: #333;'>
                "{text[:500]}{'...' if len(text) > 500 else ''}"
            </div>
        </div>
        """

    html += "</div>"
    return html


def process_question(question: str, history: List) -> Tuple[List, str]:
    """
    Process user question and return response with sources.

    Args:
        question: User's question.
        history: Chat history.

    Returns:
        Tuple of (updated_history, formatted_sources).
    """
    global pipeline

    if not question or not question.strip():
        return history, "<p><i>Please enter a question.</i></p>"

    if pipeline is None:
        error_msg = (
            "❌ System not initialized. Please wait for initialization to complete."
        )
        history.append((question, error_msg))
        return history, "<p><i>No sources available.</i></p>"

    try:
        logger.info(f"Processing question: {question}")

        # Get response from RAG pipeline
        response = pipeline.query(question)

        # Extract answer and sources
        answer = response.get("answer", "No answer generated.")
        sources = response.get("sources", [])

        logger.info(f"Generated answer with {len(sources)} sources")

        # Update chat history
        history.append((question, answer))

        # Format sources for display
        sources_html = format_sources(sources)

        return history, sources_html

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        error_msg = f"❌ Error: {str(e)}"
        history.append((question, error_msg))
        return history, "<p><i>Error retrieving sources.</i></p>"


def clear_conversation():
    """Clear chat history and sources."""
    logger.info("Clearing conversation")
    return [], "<p><i>Conversation cleared. Ask a new question!</i></p>"


def create_interface():
    """Create and configure the Gradio interface."""

    # Custom CSS for better styling
    custom_css = """
    #chatbot {
        height: 500px;
    }
    #sources-box {
        max-height: 400px;
        overflow-y: auto;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        color: #666;
    }
    """

    # Create Gradio interface
    with gr.Blocks(
        title="CrediTrust Financial - Complaint Analysis Chatbot"
    ) as demo:

        # Header
        gr.Markdown(
            """
            # 🏦 CrediTrust Financial - Complaint Analysis Chatbot
            
            Welcome! This AI-powered chatbot helps you analyze customer complaints across our financial products:
            **Credit Cards**, **Personal Loans**, **Savings Accounts**, and **Money Transfers**.
            
            ### How to use:
            1. Type your question in the text box below
            2. Click **Submit** or press Enter
            3. Review the AI-generated answer and source documents
            4. Use **Clear** to start a new conversation
            
            ### Example questions:
            - "What are the main complaints about credit cards?"
            - "Why are customers unhappy with personal loans?"
            - "What issues do people report with money transfers?"
            - "Compare complaint trends across different products"
            """
        )

        # Initialization status
        status_msg = gr.Textbox(
            label="System Status",
            value="Initializing...",
            interactive=False,
            show_label=True,
        )

        # Chat interface
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    elem_id="chatbot",
                    show_label=True,
                    avatar_images=(None, "🤖"),
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about customer complaints...",
                        lines=2,
                        show_label=True,
                        scale=4,
                    )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Submit", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

        # Sources display
        gr.Markdown("## 📚 Source Documents")
        sources_box = gr.HTML(
            label="Retrieved Sources",
            value="<p><i>Sources will appear here after you submit a question.</i></p>",
            elem_id="sources-box",
        )

        # Footer
        gr.Markdown(
            """
            <div class="footer">
                <hr>
                <p>🔒 For internal use only | Built with ❤️ by the CrediTrust AI Team</p>
                <p><i>Powered by RAG (Retrieval-Augmented Generation) technology</i></p>
            </div>
            """
        )

        # Event handlers
        submit_btn.click(
            fn=process_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, sources_box],
        ).then(
            fn=lambda: "",  # Clear input after submission
            inputs=None,
            outputs=question_input,
        )

        question_input.submit(
            fn=process_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, sources_box],
        ).then(
            fn=lambda: "",  # Clear input after submission
            inputs=None,
            outputs=question_input,
        )

        clear_btn.click(
            fn=clear_conversation, inputs=None, outputs=[chatbot, sources_box]
        )

        # Initialize pipeline on load
        demo.load(fn=initialize_pipeline, inputs=None, outputs=status_msg)

    return demo


def main():
    """Main function to launch the application."""
    logger.info("Starting Gradio application...")

    # Create interface
    demo = create_interface()

    # Launch application
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True,
    )


if __name__ == "__main__":
    main()
