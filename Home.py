import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import openai
# Load environment variables from .env file
load_dotenv()
# IMPORTANT: Remember to create a .env variable containing: OPENAI_API_KEY=sk-xyz where xyz is your key

# Access the API key from the environment variable
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
os.environ['COHERE_API_KEY'] = os.environ.get("COHERE_API_KEY")
# os.environ['ANTHROPIC_API_KEY'] = os.environ.get("ANTHROPIC_API_KEY")
openai.api_key  = os.getenv("OPENAI_API_KEY")
qdrant_url  = os.getenv('QDRANT_URI')
qdrant_api_key  = os.getenv('QDRANT_API_KEY')

from st_pages import Page, show_pages, add_page_title
st.sidebar.header("Fine Tuning")

# # Optional -- adds the title and icon to sthe current page
add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles and icons
show_pages(
    [
        Page("Home.py", "Upload Document and Assign Dataset"),
        Page("pages/1_LLM.py", "Evaluate LLM Models"),


    ]
)

# Initialize doc_path with a default value
doc_path = "docs/"

# Initialize session state keys if they don't exist
if 'eval_questions' not in st.session_state:
    st.session_state['eval_questions'] = []
if 'eval_answers' not in st.session_state:
    st.session_state['eval_answers'] = []


# Check if the user wants to use the default document or upload their own
st.header('Document Selection')
document_option = st.radio("Choose your document source", ('Upload a file', 'Use default test document'))

if document_option == 'Upload a file':
    st.session_state['eval_questions'] = [""]
    st.session_state['eval_answers'] = [""]
    # Allow multiple files to be uploaded including pdf, csv, doc, docx, ppt, pptx
    uploaded_files = st.file_uploader("Choose files", type=['pdf', 'csv', 'docx', 'pptx'], accept_multiple_files=True)
    if uploaded_files:
        # Ensure the 'uploaded_docs' directory exists before saving the files
        upload_dir = "uploaded_docs"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        # Save the uploaded files and collect their paths
        for uploaded_file in uploaded_files:
            with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Update session state with the directory name of uploaded documents
        st.session_state['doc_path'] = upload_dir

        # User input for eval_questions and eval_answers
        st.subheader('Provide Evaluation Questions and Answers')

        data = {
            'Questions': st.session_state['eval_questions'],
            'Ground Truth': st.session_state['eval_answers']
        }
        qa_df = pd.DataFrame(data)
        edited_qa_df = st.data_editor(data, num_rows="dynamic", use_container_width=True, hide_index=True)

        eval_questions_list = edited_qa_df['Questions']
        eval_answers_list = edited_qa_df['Ground Truth']

        if st.button("Save eval Q&As"):
            # Check if the number of questions matches the number of answers
            st.session_state['eval_questions'] = eval_questions_list
            st.session_state['eval_answers'] = eval_answers_list
            st.success("Evaluation questions and answers saved successfully!")

else:
    # Use the default document
    doc_path = "docs"
    st.write("Using the default document: Constitution.pdf")

    # Default eval_questions and eval_answers
    eval_questions = [
        "هل شـــهد أداء الماليـــة العامـــة فـــي عـــام 2022م ارتفاعـــاً ؟",
        "ما فائض الميزانية لعام ٢٠٢٢؟",
        "ما المقارنة الأداء الفعلي لعام 2022 مع الميزانية؟",
        "ما  النتائج اﻟﻤﺎﻟﻴﺔ  ﻟﻠﻌﺎم  2022 لمجموعة تداول السعودية؟ ",
        "النفقات الفعلية على مستوى القطاعات في القطاع العسكري",
    ]
    eval_answers = [
        "نعم، شهد أداء المالية العامة في عام 2022م ارتفاعاً في إجمالي الإيرادات بحوالي 213 مليون عن الميزانية المعتمدة.",
        "فائض الميزانية لعام 2022 بلغ حوالي 104 مليار ريال، وهو أعلى من المقدر في الميزانية والذي بلغ حوالي 90 مليار ريال.",
        "الأداء الفعلي للميزانية لعام 2022 حقق فائضاً بحوالي 104 مليار ريال، ما يعادل حوالي 2.5% من الناتج المحلي الإجمالي. هذا الفائض أعلى من المقدر في الميزانية والذي بلغ حوالي 90 مليار ريال. يُشار إلى أن مبالغ الفوائض المحققة في الميزانية ستوجه وفق آلية التعامل مع الفوائض لتعزيز الاحتياطي الحكومي ودعم الصناديق الوطنية والنظر في إمكانية تعجيل تنفيذ بعض البرامج والمشاريع الاستراتيجية ذات البعد الاقتصادي والاجتماعي بما يحقق النمو الاقتصادي المستدام وبما يضمن المحافظة على استدامة واستقرار المركز المالي للدولة.",
        """لنتائج المالية لمجموعة تداول السعودية للعام 2022 كانت كالتالي:




ارتفعت المصاريف التشغيلية بنسبة 16.6% على أساس سنوي لتصل إلى 644.3 مليون ريال سعودي مقارنة بـ 552.5 مليون ريال سعودي في العام السابق 2021. ويرجع ذلك إلى الارتفاع في تكاليف الرواتب بسبب الزيادة في أعداد الموظفين.




انخفضت الأرباح قبل الفوائد والضرائب والاستهلاك والإطفاء (EBITDA) بنسبة 26.9% على أساس سنوي لتصل إلى 490.3 مليون ريال سعودي مقارنة بـ 670.6 مليون ريال سعودي في العام السابق 2021. ويرجع ذلك إلى انخفاض الإيرادات التشغيلية للمجموعة مقابل النمو في المصاريف التشغيلية للمجموعة.




انخفض إجمالي الربح للمجموعة بنسبة 18.0% على أساس سنوي ليصل إلى 683.7 مليون ريال سعودي مقارنة بـ 834.3 مليون ريال سعودي في العام السابق 2021. ويرجع ذلك إلى الانخفاض في الإيرادات التشغيلية مقابل النمو في المصاريف التشغيلية للمجموعة.


""",
        "من خلال تحليل البيانات المقدمة، يمكننا رؤية تطور النفقات الفعلية على مستوى القطاع العسكري خلال السنوات الثلاث الماضية. في عام 2020، بلغت النفقات 204 مليار ريال، بينما في عام 2021، بلغت النفقات 202 مليار ريال. وقد تم تخصيص 171 مليار ريال للنفقات في ميزانية عام 2022، ولكن النفقات الفعلية في هذا العام بلغت 228 مليار ريال..",
    ]

    # Assign the default questions and answers to the state
    st.session_state['eval_questions'] = eval_questions
    st.session_state['eval_answers'] = eval_answers
    st.session_state['doc_path'] = doc_path

# Display eval questions and answers if available
if st.session_state.get('eval_questions') and st.session_state.get('eval_answers'):
    st.subheader('Saved Evaluation Questions and Answers')
    # Convert eval_questions and eval_answers to a DataFrame and display it
    eval_qa_df = pd.DataFrame({
        'Questions': st.session_state['eval_questions'],
        'Ground Truth': st.session_state['eval_answers']
    })
    st.dataframe(eval_qa_df, use_container_width=True , hide_index=True)
    if len(eval_qa_df["Questions"]) >= 4:
        st.subheader('Proceed to one of the tabs on the left to perform Evaluations')
        st.page_link("pages/1_LLM.py", label="LLM")


    else:
        st.warning('Please add at least 4 rows of data for evaluation')

else:
    st.header('No evaluation questions and answers provided.')
