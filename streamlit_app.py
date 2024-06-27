import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = ''  # this means "default" is selected as project for tracing
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_TRACING_V2 = "true"

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

# Sidebar information
st.sidebar.title("Developed by Abhishek Vidhate")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/abhishek-vidhate-21smdb/)")
st.sidebar.markdown("[GitHub](https://github.com/Abhishekvidhate)")

# Main app UI
st.title("Resume Optimizer/Builder")
st.write("Hello, please upload your resume/CV and job posting's URL link,"
         " so I can optimize your resume according to the job posting.")

uploaded_file = st.file_uploader("Upload your resume/CV (PDF)", type=["pdf"])
job_url = st.text_input("Enter job posting URL")
submit_button = st.button("Submit")

def process_resume(uploaded_file, job_url):
    # Save the uploaded file to a temporary location
    temp_file_path = "data/temp_resume.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process the CV
    cv_loader = LLMSherpaFileLoader(file_path=temp_file_path, new_indent_parser=True, apply_ocr=True, strategy="chunks")
    cv_content = cv_loader.load()

    # Load and process the job posting
    web_loader = WebBaseLoader(job_url)
    job_content = web_loader.load()

    # Define the prompt templates
    similarities_prompt_template = PromptTemplate(
        input_variables=["cv_content", "job_content"],
        template="""
        I have two documents: one is a user's CV and the other is a job posting. 
        I want to identify the similarities between them in terms of relevant skills, experiences, 
        and keywords that match the job requirements. Here are the documents:

        User's CV:
        {cv_content}

        Job Posting:
        {job_content}

        Analyze both documents and provide a detailed comparison highlighting the similarities. 
        Specifically, identify the key skills and experiences from the user's CV that match the job requirements. 
        Also, suggest any modifications to the CV to better align it with the job posting.
        """
    )

    similarities_prompt = similarities_prompt_template.format(cv_content=cv_content, job_content=job_content)
    similarity_chain = similarities_prompt_template | llm
    similarities = similarity_chain.invoke({"cv_content": cv_content, "job_content": job_content})

    new_cv_prompt_template = PromptTemplate(
        input_variables=["cv_content", "job_content", "similarities"],
        template="""
        System: You are an AI assistant that helps users by updating their CVs based on job postings and identified similarities. 
        Your task is to create new CV content that incorporates relevant skills and experiences from the job posting into the user's CV. 
        
        Here are the documents and the identified similarities:

        User's CV:
        {cv_content}

        Job Posting:
        {job_content}

        Similarities:
        {similarities}

        Please create a new version of the user's CV that highlights the relevant skills and experiences from the job posting. 
        Ensure that the new CV content is tailored to match the job requirements.
        """
    )

    new_cv_chain = new_cv_prompt_template | llm
    new_cv_content = new_cv_chain.invoke({
        "cv_content": cv_content,
        "job_content": job_content,
        "similarities": similarities
    })

    ats_prompt_template = PromptTemplate(
        input_variables=["new_cv_content"],
        template="""
        System: You are an AI assistant that helps users format their CVs to be ATS-friendly. 
        Your task is to convert the new CV content into a format that is easily readable by Applicant Tracking Systems. 
        Important note, only provide the optimized content and no extra text generated output from your side, 
        as the content is later going to be used to generate PDF file, so prefer using bullet points for markdown components and also use etc resume pdf components for the markdown content to be PDF-file ready. Here is the new CV content:

        {new_cv_content}

        Please format this CV content to ensure it meets ATS requirements, 
        including simple layout, keyword optimization, and clear section headers.
        """
    )

    ats_chain = ats_prompt_template | llm | StrOutputParser()
    ats_friendly_cv_response = ats_chain.invoke({"new_cv_content": new_cv_content})

    # Function to create a PDF from the ATS-friendly CV content
    def create_pdf(cv_content, file_path):
        c = canvas.Canvas(file_path, pagesize=letter)
        c.drawString(100, 750, "Resume / CV")
        text = c.beginText(50, 730)
        text.setFont("Helvetica", 12)
        for line in cv_content.split('\n'):
            text.textLine(line)
        c.drawText(text)
        c.save()

    # Path to save the PDF
    pdf_path = "data/new_cv.pdf"

    # Create the PDF
    create_pdf(ats_friendly_cv_response, pdf_path)

    st.success("Resume optimization complete!")

    st.write("Here is the modified resume. This is ATS optimized, precisely created according to the job/internship posting:")
    st.markdown(f"```\n{ats_friendly_cv_response}\n```")

    with open(pdf_path, "rb") as file:
        st.download_button(label="Download Resume", data=file, file_name="optimized_resume.pdf", mime="application/pdf")

    # Define the cover letter/why hire me prompt template
    cover_letter_prompt_template = PromptTemplate(
        input_variables=["cv_content", "job_content", "similarities"],
        template="""
        System: You are an AI assistant that helps users create a cover letter based on their CV and job posting. 
        Your task is to create a cover letter or "Why Hire Me?" message within 1000 characters, 
        focusing on the key skills and experiences that match the job requirements. 
        
        Here are the documents and the identified similarities:

        User's CV:
        {cv_content}

        Job Posting:
        {job_content}

        Similarities:
        {similarities}

        Please create a cover letter or "Why Hire Me?" 
        message that highlights the relevant skills and experiences from the user's CV that match the job requirements.
        """
    )

    cover_letter_chain = cover_letter_prompt_template | llm | StrOutputParser()
    cover_letter_response = cover_letter_chain.invoke({
        "cv_content": cv_content,
        "job_content": job_content,
        "similarities": similarities
    })

    st.write("Here's a bonus from me: Cover Letter / Why Hire Me?")
    st.text_area("Cover Letter / Why Hire Me?", cover_letter_response, height=200)

if submit_button and uploaded_file and job_url:
    process_resume(uploaded_file, job_url)