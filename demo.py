import gradio as gr
import llm  # Assuming 'llm' is the module name for the code you provided
from pathlib import Path

# Assuming llm contains the VectorDB, read_txt, read_pdf, read_docx, and process_and_add_files functions/classes

# Initialize the VectorDB
vector_db = llm.VectorDB()

# Define a function to handle file uploads and process them
def process_files(files):
    file_paths = []
    for file in files:
        # Save the uploaded file to a temporary location
        temp_file_path = Path(file.name)
        temp_file_path.write_bytes(file.read())
        file_paths.append(str(temp_file_path))
    
    # Process and add files to the vector database
    llm.process_and_add_files(vector_db, file_paths)
    
    return "Files processed and added to the database."

# Define a function to handle queries
def query_db(query):
    results = vector_db.query_texts(query)
    return results

# Create the Gradio interface
iface = gr.Interface(
    fn=query_db,
    inputs="text",
    outputs="text",
    examples=["example query"]
)

# Add a file upload interface
file_upload = gr.Interface(
    fn=process_files,
    inputs=gr.inputs.File(file_count="multiple"),
    outputs="text"
)

# Combine the interfaces into a single app
demo = gr.TabbedInterface([iface, file_upload], ["Query DB", "Upload Files"])

# Launch the app
if __name__ == "__main__":
    demo.launch()
