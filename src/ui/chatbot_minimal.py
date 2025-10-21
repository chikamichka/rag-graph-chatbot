"""
Enhanced IoT Documentation Chatbot with Modern UI
Using Gradio 3.50.0 with custom styling
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag.vector_store import VectorStore
from graph.knowledge_graph import KnowledgeGraph
from rag.hybrid_retriever import HybridRetriever

print("="*70)
print(" 🤖 INITIALIZING CHATBOT")
print("="*70)

# Initialize components
print("\n1. Loading retrieval system...")
vector_store = VectorStore("./data/embeddings", "iot_docs")
knowledge_graph = KnowledgeGraph()
retriever = HybridRetriever(vector_store, knowledge_graph, 0.6, 0.4)

print("2. Loading LLM...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, trust_remote_code=True
)
device = "mps" if torch.backends.mps.is_available() else "cpu"
if device == "mps":
    model.to(device)

print(f"✅ Chatbot ready on {device}!\n")
print("="*70)

def chat(message, history):
    """Enhanced chat function with better formatting"""
    # Retrieve context
    results = retriever.retrieve(message, top_k=3)
    context = "\n".join([r['content'][:300] for r in results])
    
    # Build prompt
    prompt = f"""Answer this question about IoT/Edge Computing based on context:

Context: {context}

Question: {message}

Answer:"""
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    if device == "mps":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean up response - remove any meta-commentary or instructions
    response = response.strip()
    
    # Stop at common separators that indicate meta-text
    separators = ['\n\nYou are', '\n\n---', '\n\nThis response', '\n\nNote:', '\n\nRemember']
    for sep in separators:
        if sep in response:
            response = response.split(sep)[0].strip()
    
    return response

# Custom CSS for modern look with better contrast
custom_css = """
#component-0 {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

#chatbot-header {
    text-align: center;
    padding: 40px 20px 20px;
    background: rgba(255, 255, 255, 0.98);
    border-radius: 20px 20px 0 0;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
}

#chatbot-title {
    font-size: 2.5em;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
}

#chatbot-desc {
    font-size: 1.1em;
    color: #475569;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 500;
}

.chatbot {
    background: #1e293b !important;
    border-radius: 0 0 20px 20px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.15) !important;
    border: none !important;
}

.message-wrap {
    padding: 20px !important;
}

.message.user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4) !important;
    font-size: 1em !important;
}

.message.bot {
    background: transparent !important;
    color: #e2e8f0 !important;
    border: none !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 14px 18px !important;
    box-shadow: none !important;
    font-size: 1em !important;
}

/* Fix for example buttons container */
.examples {
    padding: 20px !important;
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 16px !important;
    margin-top: 20px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
}

.examples > label {
    font-weight: 600 !important;
    color: #1e293b !important;
    margin-bottom: 12px !important;
    font-size: 1.1em !important;
    display: block !important;
}

/* Example buttons styling */
button.secondary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    font-size: 0.95em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

button.secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

/* Input textarea styling */
.input-box textarea,
textarea {
    background-color: #334155 !important;
    border: 2px solid #475569 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 1em !important;
    color: white !important;
    transition: all 0.3s ease !important;
}

.input-box textarea::placeholder,
textarea::placeholder {
    color: #94a3b8 !important;
}

.input-box textarea:focus,
textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    background-color: #3f4d63 !important;
}

/* Clear button styling */
.clear-btn button,
button:has(.trash-icon) {
    background: #ef4444 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.clear-btn button:hover {
    background: #dc2626 !important;
    transform: translateY(-2px) !important;
}

/* Send button styling */
.submit-btn button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

.submit-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

#footer-info {
    text-align: center;
    padding: 20px;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    margin-top: 20px;
    color: #475569;
    font-size: 0.9em;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.footer-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 5px 4px;
}

.footer-tip {
    margin-top: 12px;
    font-size: 0.95em;
    color: #64748b;
    font-weight: 500;
}
"""

# Create UI with custom header
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    # Custom header
    gr.HTML("""
        <div id="chatbot-header">
            <div id="chatbot-title">IoT Documentation Assistant</div>
            <div id="chatbot-desc">
                Your intelligent guide to IoT protocols, edge computing, security, and deployment.
                Powered by Hybrid RAG combining vector search and knowledge graphs.
            </div>
        </div>
    """)
    
    # ChatInterface
    chat_interface = gr.ChatInterface(
        fn=chat,
        examples=[
            "What is edge computing?",
            "Explain MQTT protocol",
            "How to secure IoT devices?",
            "What are deployment best practices?",
            "Compare CoAP and MQTT protocols",
            "What are the benefits of fog computing?"
        ],
        retry_btn=None,
        undo_btn=None,
        clear_btn="🗑️ Clear Chat",
        submit_btn="Send 📤",
        chatbot=gr.Chatbot(
            height=500,
            show_label=False
        )
    )
    
    # Footer
    gr.HTML("""
        <div id="footer-info">
            <p>
                <span class="footer-badge">Qwen 2.5 1.5B</span>
                <span class="footer-badge">Hybrid RAG</span>
                <span class="footer-badge">Real-time Retrieval</span>
            </p>
            <p class="footer-tip">
                💡 Ask technical questions about IoT protocols, architectures, security, and more!
            </p>
        </div>
    """)

if __name__ == "__main__":
    print("\n🚀 Launching enhanced UI at http://localhost:7860\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)