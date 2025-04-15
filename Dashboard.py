import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="Interactive Learning Guide", layout="wide")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "graph"
if "expanded" not in st.session_state:
    st.session_state.expanded = {
        "data": False,
        "python": False,
        "guidance": False
    }
# Add template state variables
if "template_sections" not in st.session_state:
    st.session_state.template_sections = {}
if "template_created" not in st.session_state:
    st.session_state.template_created = False
if "template_selections" not in st.session_state:
    st.session_state.template_selections = {
        section: {
            "selected": False,
            "levels": {level: False for level in ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"]},
            "subsections": {}
        } for section in ["Load Data", "Data Cleaning", "Analysis", "Visualization"]
    }
if "guidance_section" not in st.session_state:
    st.session_state.guidance_section = "None"
if "template_name" not in st.session_state:
    st.session_state.template_name = "My Custom Cheatsheet"
if "reset_guidance" not in st.session_state:
    st.session_state.reset_guidance = False

# =============================================================================
# Navigation Functions
# =============================================================================
def show_data_page():
    st.session_state.page = "data"

def show_graph_page():
    st.session_state.page = "graph"

def toggle_node(node):
    st.session_state.expanded[node] = not st.session_state.expanded[node]

def show_template_page():
    st.session_state.page = "template"

# =============================================================================
# Sidebar Configuration
# =============================================================================
# Main navigation buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Knowledge Graph", use_container_width=True):
        show_graph_page()
with col2:
    if st.button("Create Template", use_container_width=True):
        show_template_page()

# Python topics dropdown
st.sidebar.markdown("## Python Topics")
python_topic = st.sidebar.selectbox(
    "Select a Python topic:",
    ["None", "Basics", "Loops", "Functions", "Classes"]
)

# Function to handle guidance section change
def on_guidance_change():
    section = st.session_state.guidance_section
    if section == "Step 1 - Load Data":
        st.session_state.page = "data"
    elif section == "Step 2 - Initial Data Exploration":
        st.session_state.page = "step2_initial_exploration"
    elif section == "Step 3 - Defining Key Questions":
        st.session_state.page = "step3_key_questions"
    elif section == "Step 4 - Data Cleaning":
        st.session_state.page = "step4_data_cleaning"
    elif section == "Step 5 - EDA":
        st.session_state.page = "step5_eda"
    elif section == "Step 6 - Visualization":
        st.session_state.page = "step6_visualization"
    elif section == "Step 7 - Statistical Analysis":
        st.session_state.page = "step7_statistical_analysis"
    elif section == "Step 8 - Final Project":
        st.session_state.page = "step8_final_project"

    st.session_state.expanded["data"] = True
    st.session_state.expanded["python"] = True
    st.session_state.expanded["guidance"] = True

if "reset_guidance" not in st.session_state:
    st.session_state.reset_guidance = False

# Project Guidance dropdown
st.sidebar.markdown("## Project Guidance")
st.sidebar.selectbox(
    "Select a guidance section:",
    ["None", "Step 1 - Load Data", "Step 2 - Initial Data Exploration", "Step 3 - Defining Key Questions", 
     "Step 4 - Data Cleaning", "Step 5 - EDA", "Step 6 - Visualization", 
     "Step 7 - Statistical Analysis", "Step 8 - Final Project"],
    key="guidance_section",
    on_change=on_guidance_change
)

# =============================================================================
# Data Loading
# =============================================================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv(
            "Underlying_Cause_of_Death_2018-2022_Single_Race.txt",
            delimiter='\t',
            encoding='latin1'
        )
    except Exception as e:
        st.sidebar.warning(f"Could not load actual data file: {e}")
        # Create fallback sample data (without Year)
        import numpy as np
        return pd.DataFrame({
            'State': ['Alabama', 'Alaska', 'Arizona', 'California', 'Florida'],
            'Deaths': np.random.randint(100, 1000, 5),
            'Population': [5074296, 733583, 7200000, 39500000, 21480000]
        })

data = load_data()

# =============================================================================
# Custom CSS
# =============================================================================
st.title("📚 Data Analysis Learning Guide")
st.markdown("""
<style>
    .centered-button {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    .arrow-down {
        text-align: center;
        font-size: 20px;
        color: #888;
        margin: 5px 0;
    }
    .stButton button {
        background-color: #f0f7ff;
        border: 1px solid #d0e0ff;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #e0f0ff;
    }
    .stButton button {
        background-color: #eaf4ff !important;
        border: 1px solid #b3d1f2 !important;
        border-radius: 10px;
        color: #003366;
        font-weight: 600;
    }
    code {
        font-size: 1.1rem !important;
        font-weight: 500;
        background-color: #f6f8fa;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .template-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #4CAF50;
    }
    .template-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .cheatsheet {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 20px;
        margin-top: 20px;
    }
    .section-title {
        font-weight: bold;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
        margin-bottom: 10px;
    }
    .subsection {
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Template Content (Abbreviated sample)
# =============================================================================
template_content = {
    "Load Data": {
        "🟢 Beginner": {
            "read_csv()": {
                "description": "Load a CSV file into a pandas DataFrame",
                "code": 'df = pd.read_csv("filename.csv")',
                "example": 'df = pd.read_csv("Underlying_Cause_of_Death_2018-2022_Single_Race.txt", encoding="latin1", delimiter="\\t")',
                "output": "Returns a DataFrame with columns like State, Deaths, Population"
            },
            "path.exists()": {
                "description": "Check if a file exists before loading it",
                "code": 'import os\nif os.path.exists("filename.csv"):\n    df = pd.read_csv("filename.csv")',
                "example": 'import os\nif os.path.exists("data.csv"):\n    df = pd.read_csv("data.csv")\nelse:\n    print("File not found!")',
                "output": "If file exists: DataFrame loaded successfully\nIf file doesn’t exist: \"File not found!\""
            }
        },
        "🟡 Intermediate": {
            "read_csv() with custom headers": {
                "description": "Load a CSV file with custom column names",
                "code": 'pd.read_csv("filename.csv", header=None, names=["Col1", "Col2", "Col3"])',
                "example": 'df = pd.read_csv("data.csv", header=None, names=["State", "Code", "Cause", "Deaths"])',
                "output": "Returns a DataFrame with custom column names"
            },
            "read_csv() with usecols": {
                "description": "Load only specific columns from a CSV file",
                "code": 'pd.read_csv("filename.csv", usecols=["Column1", "Column2"])',
                "example": 'df = pd.read_csv("data.csv", usecols=["State", "Deaths"])',
                "output": "Returns a DataFrame with only selected columns"
            },
            "read_excel()": {
                "description": "Load data from an Excel file",
                "code": 'pd.read_excel("filename.xlsx")',
                "example": 'df = pd.read_excel("data.xlsx", sheet_name="2022 Data")',
                "output": "Returns a DataFrame from the specified Excel sheet"
            }
        },
        "🔴 Advanced": {
            "Using Paths": {
                "description": "Use proper paths to load files from different directories",
                "code": 'import os\nfile_path = os.path.join("folder", "subfolder", "filename.csv")\ndf = pd.read_csv(file_path)',
                "example": r'# Windows example with raw string\ndf = pd.read_csv(r"C:\Users\Name\Documents\data.csv")',
                "output": "Loads the file using a proper path across operating systems"
            },
            "Error Handling": {
                "description": "Add error handling when loading files",
                "code": 'try:\n    df = pd.read_csv("filename.csv")\nexcept FileNotFoundError:\n    print("File not found!")',
                "example": 'try:\n    df = pd.read_csv("data.csv")\nexcept FileNotFoundError:\n    print("File not found, using sample data")\n    df = pd.DataFrame({"Column1": [1, 2, 3]})',
                "output": "Handles errors gracefully with fallback data"
            }
        }
    },
    "Data Cleaning": {
        "🟢 Beginner": {
            "Handle Missing Values": {
                "description": "Fill or drop missing values in your dataset",
                "code": 'df.fillna(0)  # Fill with zeros\ndf.dropna()  # Drop rows with any NaN',
                "example": 'clean_df = df.fillna({"Deaths": 0, "State": "Unknown"})',
                "output": "Returns DataFrame with missing values filled or dropped"
            },
            "Rename Columns": {
                "description": "Rename columns to make them more readable",
                "code": 'df.rename(columns={"old_name": "new_name"})',
                "example": 'df.rename(columns={"ST": "State", "DT": "Deaths"})',
                "output": "Returns DataFrame with renamed columns"
            }
        },
        "🟡 Intermediate": {
            "Data Type Conversion": {
                "description": "Convert columns to appropriate data types",
                "code": 'df["column"] = df["column"].astype("int")',
                "example": 'df["Deaths"] = df["Deaths"].astype("int")',
                "output": "Returns DataFrame with columns converted to appropriate data types"
            },
            "Filter Rows": {
                "description": "Select specific rows based on conditions",
                "code": 'df[df["column"] > value]',
                "example": 'high_deaths = df[df["Deaths"] > 1000]',
                "output": "Returns filtered DataFrame with rows meeting the condition"
            }
        },
        "🔴 Advanced": {
            "Apply Functions": {
                "description": "Apply custom functions to transform data",
                "code": 'df["new_column"] = df["column"].apply(lambda x: your_function(x))',
                "example": 'df["DeathRate"] = df.apply(lambda row: row["Deaths"] / row["Population"] * 100000, axis=1)',
                "output": "Returns DataFrame with a new column of transformed data"
            },
            "Group and Aggregate": {
                "description": "Group data and calculate aggregations",
                "code": 'df.groupby("column").agg({"numeric_col": ["mean", "sum"]})',
                "example": 'state_summary = df.groupby("State").agg({\n    "Deaths": ["sum", "mean", "max"]\n})',
                "output": "Returns grouped DataFrame with aggregated statistics"
            }
        }
    },
    "Analysis": {
        "🟢 Beginner": {
            "Basic Statistics": {
                "description": "Calculate basic statistical measures",
                "code": 'df.describe()  # Summary statistics\n df["Deaths"].mean()  # Average\n df["Deaths"].max()  # Maximum value',
                "example": 'stats = df["Deaths"].describe()\navg = df["Deaths"].mean()',
                "output": "Returns statistical summary or specific measure"
            },
            "Value Counts": {
                "description": "Count unique values in a column",
                "code": 'df["column"].value_counts()',
                "example": 'state_counts = df["State"].value_counts()',
                "output": "Returns counts of unique values"
            }
        },
        "🟡 Intermediate": {
            "Correlation Analysis": {
                "description": "Find relationships between numerical variables",
                "code": 'df.corr()  # Correlation matrix\n df.plot.scatter(x="Population", y="Deaths")',
                "example": 'correlation = df[["Deaths", "Population"]].corr()',
                "output": "Returns a correlation matrix and scatter plot"
            },
            "Pivot Tables": {
                "description": "Create summary tables of your data",
                "code": 'pd.pivot_table(df, values="Deaths", index="State", aggfunc="sum")',
                "example": 'death_pivot = pd.pivot_table(df, values="Deaths", index="State", aggfunc="sum")',
                "output": "Returns a pivot table summarizing deaths by state"
            }
        },
        "🔴 Advanced": {
            "Time Series Analysis (Not Applicable)": {
                "description": "The dataset does not include any time or date column.",
                "code": "# Not applicable",
                "example": "# Time series analysis is not available for this dataset.",
                "output": "N/A"
            },
            "Statistical Tests": {
                "description": "Perform statistical hypothesis tests",
                "code": 'from scipy import stats\nstats.ttest_ind(group1, group2)',
                "example": 'from scipy import stats\n# Example: Compare deaths in two subsets\nsubset1 = df[df["State"] == "Alabama"]["Deaths"]\nsubset2 = df[df["State"] == "Florida"]["Deaths"]\nresult = stats.ttest_ind(subset1, subset2)',
                "output": "Returns a t-statistic and a p-value"
            }
        }
    },
    "Visualization": {
        "🟢 Beginner": {
            "Line Chart": {
                "description": "Plot numeric values – here we use State as a categorical x-axis.",
                "code": 'df_sorted = data.sort_values("State")\ndf_sorted.plot(kind="line", x="State", y="Deaths", figsize=(10, 6))',
                "example": 'df_sorted = data.sort_values("State")\ndf_sorted.plot(kind="line", x="State", y="Deaths", figsize=(10,6), title="Deaths by State")',
                "output": "Returns a line chart of deaths by state"
            },
            "Bar Chart": {
                "description": "Compare values across categories",
                "code": 'top_states = data.groupby("State")["Deaths"].sum().nlargest(10)\ntop_states.plot(kind="bar", figsize=(10, 6))',
                "example": 'top_states = data.groupby("State")["Deaths"].sum().nlargest(10)\ntop_states.plot(kind="bar", figsize=(10,6), title="Top 10 States by Deaths")',
                "output": "Returns a bar chart comparing states by total deaths"
            }
        },
        "🟡 Intermediate": {
            "Multiple Charts": {
                "description": "Create multiple charts in one figure",
                "code": 'import matplotlib.pyplot as plt\nfig, axes = plt.subplots(1, 2, figsize=(12, 5))\ndata.groupby("State")["Deaths"].sum().plot(ax=axes[0], kind="line")\ndata.groupby("State")["Deaths"].sum().nlargest(5).plot(ax=axes[1], kind="bar")',
                "example": 'import matplotlib.pyplot as plt\nfig, axes = plt.subplots(1, 2, figsize=(15,6))\nline_data = data.groupby("State")["Deaths"].sum().sort_index()\nbar_data = data.groupby("State")["Deaths"].sum().nlargest(5)\nline_data.plot(ax=axes[0], kind="line", title="Total Deaths by State")\nbar_data.plot(ax=axes[1], kind="bar", title="Top 5 States by Deaths")',
                "output": "Returns a figure with two charts for comparison"
            },
            "Heatmap": {
                "description": "Visualize data intensity with colors",
                "code": 'import seaborn as sns\npivot = pd.pivot_table(data, values="Deaths", index="State", aggfunc="sum")\nsns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f")',
                "example": 'import seaborn as sns\npivot = pd.pivot_table(data, values="Deaths", index="State", aggfunc="sum")\nsns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f")',
                "output": "Returns a heatmap showing total deaths by state"
            }
        },
        "🔴 Advanced": {
            "Interactive Charts": {
                "description": "Create charts that users can interact with",
                "code": 'import plotly.express as px\nfig = px.choropleth(\n    data,\n    locations="State",\n    locationmode="USA-states",\n    color="Deaths",\n    scope="usa",\n    title="Deaths by State"\n)\nfig.show()',
                "example": 'import plotly.express as px\nfig = px.choropleth(\n    data,\n    locations="State",\n    locationmode="USA-states",\n    color="Deaths",\n    scope="usa",\n    title="Deaths by State"\n)\nfig.show()',
                "output": "Returns an interactive choropleth map"
            },
            "Custom Matplotlib": {
                "description": "Create highly customized visualizations",
                "code": 'import matplotlib.pyplot as plt\nimport numpy as np\n\nfig, ax = plt.subplots(figsize=(12, 8))\nstates = np.array(sorted(data["State"].unique()))\ndeaths_by_state = [data[data["State"] == state]["Deaths"].sum() for state in states]\n\nax.scatter(range(len(states)), deaths_by_state, s=100, color="blue", alpha=0.7)\nplt.xticks(range(len(states)), states, rotation=45)\nplt.title("Total Deaths by State")\nplt.tight_layout()\nplt.show()',
                "example": 'import matplotlib.pyplot as plt\nimport numpy as np\n\nfig, ax = plt.subplots(figsize=(12, 8))\nstates = np.array(sorted(data["State"].unique()))\ndeaths_by_state = [data[data["State"] == state]["Deaths"].sum() for state in states]\n\nax.scatter(range(len(states)), deaths_by_state, s=100, color="blue", alpha=0.7)\nplt.xticks(range(len(states)), states, rotation=45)\nplt.title("Total Deaths by State")\nplt.tight_layout()\nplt.show()',
                "output": "Returns a custom scatter plot showing total deaths by state"
            }
        }
    }
}

# =============================================================================
# Main App Pages
# =============================================================================
if st.session_state.page == "graph":
    st.header("🌐 Knowledge Graph")
    st.write("This is the Knowledge Graph page. (Insert your interactive graph here.)")
    
    # Level 1: Data Analysis node
    data_expanded = st.session_state.expanded["data"]
    data_icon = "▼" if data_expanded else "▶"
    _, col2, _ = st.columns([2, 1, 2])
    with col2:
        if st.button(f"{data_icon} Data Analysis", key="data_btn", use_container_width=True):
            toggle_node("data")
            st.rerun()
    
    if data_expanded:
        st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
        _, col2, _ = st.columns([2, 1, 2])
        with col2:
            python_expanded = st.session_state.expanded["python"]
            python_icon = "▼" if python_expanded else "▶"
            if st.button(f"{python_icon} Python", key="python_btn", use_container_width=True):
                toggle_node("python")
                st.rerun()
        if python_expanded:
            st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
            _, col1, col2, _ = st.columns([1.75, 0.75, 0.75, 1.75])
            with col1:
                st.button("📘 Basics", key="basics_btn", use_container_width=True)
            with col2:
                guidance_expanded = st.session_state.expanded["guidance"]
                guidance_icon = "▼" if guidance_expanded else "▶"
                if st.button(f"{guidance_icon} Project Guidance", key="guidance_btn", use_container_width=True):
                    toggle_node("guidance")
                    st.rerun()
            if guidance_expanded:
                st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
                _, _, col2, _ = st.columns([1.75, 0.75, 0.75, 1.75])
                with col2:
                    if st.button("📂 Load Data", key="load_btn", on_click=show_data_page, use_container_width=True):
                        pass

elif st.session_state.page == "data":
    st.header("📂 Loading Data - Interactive Guide")
    if st.button("← Back to Knowledge Graph", key="back_btn"):
        st.session_state.page = "graph"
        st.session_state.reset_guidance = True
        st.rerun()
    st.subheader("Choose Your Level")
    tab1, tab2, tab3 = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"])
    with tab1:
        st.markdown("### 🟢 Beginner Level: File Loading Basics")
        with st.expander("📂 Step 1: Load a File in the Same Folder"):
            st.markdown("""
- Use `pd.read_csv()` to load a `.csv` or `.txt` file directly into a DataFrame.
- If your dataset is in the **same folder** as your script, just provide the filename.
- Example using a relative path:
            """)
            st.code('df = pd.read_csv("Underlying_Cause_of_Death_2018-2022_Single_Race.txt", encoding="latin1", delimiter="\\t")\ndf.head()', language="python")
            st.dataframe(data.head())
        with st.expander("🗂️ Folder Structure Example"):
            st.markdown("This image shows how your file and dataset should be placed in the same directory.")
            st.image(r"./Images/SameFolder.png", caption="Same folder: script + file", use_container_width=True)
        with st.expander("✅ Check Your Understanding"):
            st.markdown("""
**Scenario:**  
You have a file called `data.csv` in the **same folder** as your notebook.

👉 How would you load it using pandas?  
(Type your answer below)
            """)
            user_input = st.text_input("Your code:")
            correct_answers = [
                'pd.read_csv("data.csv")',
                "pd.read_csv('data.csv')",
                'df = pd.read_csv("data.csv")',
                "df = pd.read_csv('data.csv')"
            ]
            if user_input:
                trimmed_input = user_input.strip()
                if trimmed_input in correct_answers:
                    st.success("✅ Great job! Your syntax is correct.")
                    st.markdown("Here's a preview of sample data:")
                    fake_df = pd.DataFrame({
                        "Name": ["Alice", "Bob", "Carlos"],
                        "Age": [25, 30, 22],
                        "City": ["New York", "Chicago", "Miami"]
                    })
                    st.dataframe(fake_df)
                else:
                    st.error("❌ That's not quite right.")
                    st.markdown("🔍 Hint: Use `pd.read_csv()` and enclose the filename in quotes.")
    with tab2:
        st.markdown("### 🟡 Intermediate Data Loading Techniques")
        with st.expander("📑 Load Data with Custom Headers"):
            st.markdown("Assign your own column names if headers are missing.")
            st.code('pd.read_csv("data.csv", header=None, names=["State", "Code", "Cause", "Deaths"])', language="python")
        with st.expander("🧮 Load Specific Columns"):
            st.markdown("Load only necessary columns to save memory.")
            st.code('pd.read_csv("data.csv", usecols=["State", "Deaths"])', language="python")
        with st.expander("⚡ Optional Performance Tweaks"):
            st.markdown("""
When working with large files, use parameters such as:
- `usecols`
- `nrows`
- `dtype`
            """)
            st.code('df = pd.read_csv("data.csv", usecols=["State", "Deaths"], nrows=1000, dtype={"Deaths": "int32"})', language="python")
        with st.expander("🧾 Load Data from JSON"):
            st.markdown("Load JSON files with pandas using:")
            st.code('pd.read_json("data.json", lines=True)', language="python")
        with st.expander("📊 Load Data from Excel"):
            st.markdown("Use `read_excel()` to open Excel files.")
            st.code('pd.read_excel("data.xlsx")', language="python")
        with st.expander("📄 Load from a Specific Excel Sheet"):
            st.markdown("Specify the worksheet with the `sheet_name` parameter.")
            st.code('pd.read_excel("data.xlsx", sheet_name="2022 Data")', language="python")
    with tab3:
        st.markdown("### 🔴 Advanced Techniques: Paths, Safety, and Performance")
        with st.expander("🧭 Relative vs Absolute Paths"):
            st.markdown("""
Use relative paths like `"data/myfile.csv"` or absolute paths like `"C:/Users/Jesus/Documents/data.csv"`.
            """)
            st.image("images/relative-path-windows.png", caption="Relative vs Absolute Paths", use_container_width=True)
        with st.expander("📌 Windows Tip: Raw Strings"):
            st.markdown("Prefix file paths with `r` on Windows to avoid issues with backslashes.")
            st.code(r'df = pd.read_csv(r"C:\Users\Jesus\Desktop\data.csv", delimiter="\t", encoding="latin1")', language="python")
        with st.expander("🧩 Using os.path.join()"):
            st.markdown("Build file paths safely across OSes.")
            st.code('import os\nfile_path = os.path.join("data", "myfile.csv")\ndf = pd.read_csv(file_path)', language="python")
        with st.expander("🛡️ Error Handling"):
            st.markdown("Use try-except to handle errors during file load.")
            st.code('try:\n    df = pd.read_csv("data/myfile.csv")\nexcept FileNotFoundError:\n    st.error("File not found.")', language="python")
        with st.expander("✅ Check Your Understanding: Paths"):
            st.markdown("""
**Scenario:**  
Your script is in:  
`C:/Users/Jesus/Documents/Project/scripts/main.py`  
Data file is at:  
`C:/Users/Jesus/Documents/data/stats.csv`

How many levels up do you need to go?
            """)
            st.markdown("""
- From `scripts/` to `Project/`: `..`
- From `Project/` to `Documents/`: `..`
- Then into `data/`
            """)
            st.markdown("Final relative path:")
            st.code("../../data/stats.csv", language="python")
            user_path = st.text_input("Enter your relative path to stats.csv:")
            valid_answers = ["../../data/stats.csv", r"..\\..\\data\\stats.csv"]
            if user_path:
                if user_path.strip() in valid_answers:
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Not quite. Remember: two levels up then into the `data` folder.")
elif st.session_state.page == "step2_initial_exploration":
    st.header("📊 Step 2 – Initial Data Exploration")
    
    # Navigation back to Knowledge Graph
    if st.button("← Back to Knowledge Graph", key="back_btn_step2"):
        st.session_state.page = "graph"
        st.session_state.reset_guidance = True
        st.rerun()
    
    st.subheader("Choose Your Level")
    tab1, tab2, tab3 = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"])
    
    # =====================================================================
    # BEGINNER: Basic Data Preview
    # =====================================================================
    with tab1:
        st.markdown("### 🟢 Beginner Level: Basic Data Preview")
        
        with st.expander("1. Display First Few Rows"):
            st.markdown("- Use `df.head()` to view the first few rows of the DataFrame. You can also specify a number, e.g., `df.head(10)` to see the first 10 rows.")
            st.code("df.head()", language="python")
            st.dataframe(data.head())
            
        with st.expander("2. Display Last Few Rows"):
            st.markdown("- Use `df.tail()` to view the last few rows of the DataFrame. Similarly, you can specify a number such as `df.tail(10)`.")
            st.code("df.tail()", language="python")
            st.dataframe(data.tail())
            
        with st.expander("3. List Columns"):
            st.markdown("- Use `df.columns` to display all the column names in the DataFrame.")
            st.code("df.columns", language="python")
            st.write(data.columns)
            
        with st.expander("4. DataFrame Dimensions"):
            st.markdown("- Use `df.shape` to check the total number of rows and columns.")
            st.code("df.shape", language="python")
            st.write(data.shape)
            
        with st.expander("5. Quick Preview by Evaluating"):
            st.markdown("""
In a Jupyter Notebook (e.g., in VS Code), simply entering `df` in a cell and running it will display a quick preview of the DataFrame—typically the first five and the last five rows, with ellipsis in between if the DataFrame is large.
            
You can also use:
- `df.head(n)` to view the first *n* rows,
- `df.tail(n)` to view the last *n* rows, and 
- `df.size` to check the total number of elements (rows × columns).
            """)
    
    # =====================================================================
    # INTERMEDIATE: Detailed Summary
    # =====================================================================
    with tab2:
        st.markdown("### 🟡 Intermediate Level: Detailed Summary")
        
        with st.expander("1. Summary Statistics"):
            st.markdown("- Use `df.describe()` to obtain basic summary statistics for numeric columns.")
            st.code("df.describe()", language="python")
            st.write(data.describe())
            
        with st.expander("2. Data Types"):
            st.markdown("- Use `df.dtypes` to see the data type for each column in the DataFrame.")
            st.code("df.dtypes", language="python")
            st.write(data.dtypes)
            
        with st.expander("3. Value Counts for 'State' Column"):
            st.markdown("- Use `df['State'].value_counts()` to count the occurrences of each unique value in the 'State' column.")
            st.code("df['State'].value_counts()", language="python")
            st.write(data['State'].value_counts())
            
        with st.expander("4. Unique Counts"):
            st.markdown("- Use `df.nunique()` to check the number of unique values per column.")
            st.code("df.nunique()", language="python")
            st.write(data.nunique())
            
        with st.expander("5. DataFrame Information"):
            st.markdown("""
- **`df.info()`** prints a concise summary of the DataFrame, including the index dtype and columns, the non-null count in each column, and the data type of each column along with memory usage.

            """)
            st.code("df.info()", language="python")
            import io
            buffer = io.StringIO()
            data.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
            
        with st.expander("6. Transposed Summary Statistics"):
            st.markdown("- Transpose the summary statistics using `df.describe().T` for easier reading when there are many columns.")
            st.code("df.describe().T", language="python")
            st.write(data.describe().T)
            
        with st.expander("7. Missing Values Check"):
            st.markdown("""
- Use **`df.isnull().sum()`** to count the number of missing (`NaN`) values in each column.
- Compare this with `df.info()`: while `df.info()` shows non-null counts, `df.isnull().sum()` gives you the exact number of missing values.
            """)
            st.code("df.isnull().sum()", language="python")
            st.write(data.isnull().sum())
    
    # =====================================================================
    # ADVANCED: Complex Exploration Techniques
    # =====================================================================
    with tab3:
        st.markdown("### 🔴 Advanced Level: Complex Exploration Techniques")
        
        with st.expander("1. Chained Filtering with Multiple Conditions"):
            st.markdown(
                "- Filter the DataFrame based on multiple conditions. For example, show rows where the 'State' is 'Alabama' and 'Deaths' is greater than 100."
            )
            st.code(
'''filtered_df = data[(data["State"] == "Alabama") & (data["Deaths"] > 100)]
filtered_df.head()''', language="python")
            st.dataframe(data[(data["State"] == "Alabama") & (data["Deaths"] > 100)])
            
        with st.expander("2. Custom Aggregations Using"):
            st.markdown(
                "- Group the data by 'State' and compute aggregate statistics (e.g., total and average of 'Deaths', average 'Population')."
            )
            st.code(
'''agg_df = data.groupby("State").agg({
    "Deaths": ["sum", "mean"],
    "Population": "mean"
})
agg_df''', language="python")
            st.write(data.groupby("State").agg({
                "Deaths": ["sum", "mean"],
                "Population": "mean"
            }))
            
        with st.expander("3. Sorting and Filtering for High Death Counts"):
            st.markdown("- Sort the DataFrame by 'Deaths' in descending order and display the top entries.")
            st.code(
'''sorted_df = data.sort_values("Deaths", ascending=False)
sorted_df.head()''', language="python")
            st.dataframe(data.sort_values("Deaths", ascending=False).head())


elif st.session_state.page == "step3_key_questions":
    st.header("Step 3 - Defining Key Questions")
    # Navigation back to Knowledge Graph
    if st.button("← Back to Knowledge Graph", key="back_btn_step2"):
        st.session_state.page = "graph"
        st.session_state.reset_guidance = True
        st.rerun()
    st.subheader("Understanding the Data")
    st.write("Your CDC dataset on underlying causes of death includes these key columns:")
    st.write("- **State**")
    st.write("- **Cause of death**")
    st.write("- **Deaths**")
    st.write("- **Population**")
    st.write("- **Crude Rate** (Note: Some values may be labeled as 'Unreliable')")
    st.write("Make sure to handle missing or non‑numeric values (for example, in the Crude Rate column) appropriately before proceeding.")
    st.write("")
    st.write("Below is an outline of specific questions you can ask about the dataset, along with the process to answer them and sample code for guidance.")

    st.subheader("Choose Your Level")

    tab_beginner, tab_intermediate, tab_advanced = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"])

    # --------------------- BEGINNER LEVEL ---------------------
    with tab_beginner:
        st.markdown("### 🟢 Beginner Level: Basic Aggregations & Descriptive Statistics")
        
        with st.expander("Which state has the highest total number of deaths?"):
            st.write("Process: Group the data by the **State** column and sum the **Deaths**. Then, sort these totals in descending order to determine which state has the highest value. You can also visualize the results with a bar chart.")
            code_text = (
                "state_deaths = data.groupby('State')['Deaths'].sum().sort_values(ascending=False)\n"
                "print(state_deaths)\n\n"
                "import matplotlib.pyplot as plt\n"
                "state_deaths.plot(kind='bar', figsize=(10, 6), title='Total Deaths by State')\n"
                "plt.ylabel('Deaths')\n"
                "plt.show()"
            )
            st.code(code_text, language="python")
        
        with st.expander("What are the summary statistics for key numeric variables?"):
            st.write("Process: Use `data.describe().T` to obtain a transposed summary of numerical features. Check the overall number of elements with `data.size`, review column data types with `data.info()`, and count missing values using `data.isnull().sum()`.")
            code_text = (
                "print(data.describe().T)\n"
                "print('Total number of elements:', data.size)\n"
                "data.info()\n"
                "print('Missing values per column:')\n"
                "print(data.isnull().sum())"
            )
            st.code(code_text, language="python")

    # --------------------- INTERMEDIATE LEVEL ---------------------
    with tab_intermediate:
        st.markdown("### 🟡 Intermediate Level: Deeper Grouping and Relationship Analysis")

        
        with st.expander("Which cause of death is responsible for the highest number of deaths overall?"):
            st.write("Process: Group the data by the **Cause of death** column, sum the **Deaths** for each cause, and then sort the results in descending order to reveal the leading causes.")
            code_text = (
                "cause_deaths = data.groupby('Cause of death')['Deaths'].sum().sort_values(ascending=False)\n"
                "print(cause_deaths.head(10))"
            )
            st.code(code_text, language="python")
        
        with st.expander("What is the relationship between a state's population and its total deaths?"):
            st.write("Process: Aggregate the data by **State** to compute the total **Deaths** and the total **Population** for each state. Then use a scatter plot to visually examine if higher state populations are associated with higher death counts.")
            code_text = (
                "state_stats = data.groupby('State').agg({'Deaths': 'sum', 'Population': 'sum'})\n"
                "print(state_stats)\n\n"
                "import matplotlib.pyplot as plt\n"
                "plt.scatter(state_stats['Population'], state_stats['Deaths'])\n"
                "plt.xlabel('Population')\n"
                "plt.ylabel('Total Deaths')\n"
                "plt.title('Population vs Total Deaths')\n"
                "plt.show()"
            )
            st.code(code_text, language="python")

    # --------------------- ADVANCED LEVEL ---------------------
    with tab_advanced:
        st.markdown("### 🔴 Advanced Level: ICD-10 Mapping and Interactive Visualization")
   
        with st.expander("Mapping ICD-10 codes to ICD chapters and visualizing data"):
            st.write("Process:")
            st.write("1. Define a dictionary mapping ICD-10 code prefixes (typically the first letter) to broader ICD chapters.")
            st.write("2. Create a new column (e.g. 'ICD Chapter') by applying a function to the 'Cause of death Code' column that extracts the first character and maps it using the dictionary.")
            st.write("3. Group the data by this new 'ICD Chapter' column to compute aggregated totals for **Deaths** and calculate the average numeric **Crude Rate** (after converting the crude rate to numeric values).")
            st.write("4. Build an interactive bar chart using Plotly Express to display total deaths and average crude rates by ICD chapter.")
            code_text = (
                "icd_mapping = {\n"
                "    'A': 'Infectious Diseases',\n"
                "    'B': 'Infectious Diseases',\n"
                "    'C': 'Neoplasms',\n"
                "    'D': 'Hematologic and Immune Disorders',\n"
                "    'E': 'Endocrine and Metabolic Diseases',\n"
                "    'F': 'Mental and Behavioural Disorders',\n"
                "    'G': 'Diseases of the Nervous System',\n"
                "    'H': 'Diseases of the Eye and Ear',\n"
                "    'I': 'Diseases of the Circulatory System',\n"
                "    'J': 'Diseases of the Respiratory System',\n"
                "    'K': 'Diseases of the Digestive System',\n"
                "    'L': 'Diseases of the Skin and Subcutaneous Tissue',\n"
                "    'M': 'Diseases of the Musculoskeletal System',\n"
                "    'N': 'Diseases of the Genitourinary System',\n"
                "    'O': 'Pregnancy Related',\n"
                "    'P': 'Perinatal Conditions',\n"
                "    'Q': 'Congenital Anomalies',\n"
                "    'R': 'Symptoms and Abnormal Findings'\n"
                "}\n\n"
                "def map_icd(code):\n"
                "    if pd.isna(code):\n"
                "        return 'Unknown'\n"
                "    letter = str(code)[0]\n"
                "    return icd_mapping.get(letter, 'Other')\n\n"
                "data['ICD Chapter'] = data['Cause of death Code'].apply(map_icd)\n"
                "chapter_deaths = data.groupby('ICD Chapter')['Deaths'].sum().reset_index()\n\n"
                "data['Crude Rate Numeric'] = pd.to_numeric(data['Crude Rate'], errors='coerce')\n"
                "chapter_crude = data.groupby('ICD Chapter')['Crude Rate Numeric'].mean().reset_index()\n"
                "chapter_summary = pd.merge(chapter_deaths, chapter_crude, on='ICD Chapter')\n\n"
                "import plotly.express as px\n"
                "fig = px.bar(chapter_summary, x='ICD Chapter', y='Deaths',\n"
                "             hover_data={'Crude Rate Numeric': ':.2f'},\n"
                "             title='Total Deaths and Average Crude Rate by ICD Chapter')\n"
                "fig.show()"
            )
            st.code(code_text, language="python")

# =============================================================================
# Step 4 – Data Cleaning (Advanced Level Section)
# =============================================================================
# This is the updated code for the Step 4 - Data Cleaning section

elif st.session_state.page == "step4_data_cleaning":
    st.header("🧹 Step 4 – Data Cleaning")
    # Back navigation button
    if st.button("← Back to Knowledge Graph", key="back_btn_step4"):
        st.session_state.page = "graph"
        st.session_state.reset_guidance = True
        st.rerun()
        
    st.subheader("Choose Your Level")
    tab_beginner, tab_intermediate, tab_advanced = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"])
    
    # -------------------------------------------------------------------------
    # BEGINNER LEVEL
    # -------------------------------------------------------------------------
    with tab_beginner:
        st.markdown("### 🟢 Beginner Level: Essential Data Cleaning")
        
        with st.expander("📊 Handle Missing Values"):
            st.markdown("""
### Missing Values

Missing values in your dataset can cause errors and skew your analysis. Pandas provides several ways to detect and handle them:
            """)
            st.code('# Check for missing values\ndf.isna().sum()', language="python")
            
            # Display actual output
            missing_vals = data.isna().sum()
            st.dataframe(missing_vals)
            
            st.markdown("""
To drop rows with missing values, use `dropna()`:
            """)
            st.code('# Drop rows with missing values\ndf_no_missing = df.dropna()\n\nprint(f"Original shape: {df.shape}")\nprint(f"After dropping NA: {df_no_missing.shape}")\n\n# Check why all rows were dropped\nprint("\\nNumber of rows with at least one missing value:")\nprint(df.isna().any(axis=1).sum())\nprint(f"Percentage: {df.isna().any(axis=1).sum() / len(df) * 100:.2f}%")', language="python")
            
            # Calculate and display the actual results with the user's data
            rows_with_na = data.isna().any(axis=1).sum()
            percentage = rows_with_na / len(data) * 100 if len(data) > 0 else 0
            
            st.code(f"""Original shape: {data.shape}
After dropping NA: {data.dropna().shape}

Number of rows with at least one missing value:
{rows_with_na}
Percentage: {percentage:.2f}%""")
            
            # Explain why all or most rows were dropped
            if data.shape[0] > 0 and data.dropna().shape[0] == 0:
                st.warning("⚠️ All rows were dropped because every row has at least one missing value in some column. This often happens with real-world datasets.")
                
                # Show a sample row with its missing values highlighted
                sample_row = data.iloc[0]
                missing_cols = sample_row.index[sample_row.isna()]
                
                st.markdown("**Example:** Here's the first row showing which columns have missing values:")
                
                # Create a DataFrame to display the row with missing values highlighted
                sample_df = pd.DataFrame([sample_row])
                st.dataframe(sample_df)
                
            st.markdown("""
Instead of dropping rows, you can fill missing values with appropriate replacements based on the column type:
            """)
            st.code('''# Fill missing values appropriately for each column type
# For numeric columns: often 0, mean, or median 
df["Deaths"].fillna(0)  

# For categorical columns: often "Unknown" or most frequent value
df["State"].fillna("Unknown")

# Check if your filling strategy worked
print("Missing values before:", df["Deaths"].isna().sum())
print("Missing values after:", df["Deaths"].fillna(0).isna().sum())''', language="python")
            
            # Show real example of before and after with the actual dataset
            if 'Deaths' in data.columns:
                missing_before = data['Deaths'].isna().sum()
                missing_after = data['Deaths'].fillna(0).isna().sum()
                
                st.code(f"""Missing values before: {missing_before}
Missing values after: {missing_after}""")
                
                # Only show examples if there are actually missing values
                if missing_before > 0:
                    # Find rows with missing Deaths
                    sample_rows = data[data['Deaths'].isna()].head(3)
                    if len(sample_rows) > 0:
                        filled_rows = sample_rows.copy()
                        filled_rows['Deaths'] = filled_rows['Deaths'].fillna(0)
                        
                        st.markdown("**Example rows with missing values:**")
                        st.dataframe(sample_rows)
                        
                        st.markdown("**After filling with zeros:**")
                        st.dataframe(filled_rows)
                    else:
                        st.info("No rows with missing values in the 'Deaths' column to display.")
                else:
                    st.info("The 'Deaths' column doesn't have any missing values in this dataset. The example shows the concept using a column that does have missing values.")
                    
                    # Find any column with missing values to demonstrate
                    cols_with_na = data.columns[data.isna().any()].tolist()
                    if cols_with_na:
                        example_col = cols_with_na[0]
                        
                        sample_rows = data[data[example_col].isna()].head(3)
                        if len(sample_rows) > 0:
                            filled_rows = sample_rows.copy()
                            
                            # Fill appropriately based on dtype
                            if np.issubdtype(data[example_col].dtype, np.number):
                                filled_rows[example_col] = filled_rows[example_col].fillna(0)
                                fill_value = "0"
                            else:
                                filled_rows[example_col] = filled_rows[example_col].fillna("Unknown")
                                fill_value = "Unknown"
                            
                            st.markdown(f"**Example using '{example_col}' column which has {data[example_col].isna().sum()} missing values:**")
                            
                            st.markdown("**Before filling:**")
                            st.dataframe(sample_rows[[example_col]])
                            
                            st.markdown(f"**After filling with {fill_value}:**")
                            st.dataframe(filled_rows[[example_col]])
        
        with st.expander("🔄 Remove Duplicates"):
            st.markdown("""
### Duplicate Rows

Duplicate data can bias your analysis by giving extra weight to certain records. Use these methods to find and remove duplicates:
            """)
            st.code('# Check for duplicates\ndf.duplicated().sum()', language="python")
            
            # Display actual output
            dup_count = data.duplicated().sum()
            st.code(f"Number of duplicate rows: {dup_count}")
            
            st.markdown("""
Remove duplicates with `drop_duplicates()`:
            """)
            st.code('# Drop duplicate rows\ndf_no_dupes = df.drop_duplicates()', language="python")
            
            # Show comparison if there are duplicates
            if dup_count > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original rows", data.shape[0])
                with col2:
                    st.metric("After dropping duplicates", data.drop_duplicates().shape[0])
            else:
                st.info("No duplicates found in this dataset.")
                
            st.markdown("""
You can also drop duplicates based on specific columns:
            """)
            st.code('# Drop duplicates based on specific columns\ndf.drop_duplicates(subset=["State", "Cause of death"])', language="python")
            
            # Show example if columns exist
            if all(col in data.columns for col in ["State", "Cause of death"]):
                before_count = data.shape[0]
                after_count = data.drop_duplicates(subset=["State", "Cause of death"]).shape[0]
                st.code(f"Rows before: {before_count}\nRows after: {after_count}")
        
        with st.expander("🔀 Basic Data Transformation"):
            st.markdown("""
### Data Transformation

Transforming data into the right format is a key part of the cleaning process.

**1. Convert data types:**
            """)
            st.code('# Convert to numeric and then to integers\ndf["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce")\ndf["Deaths"] = df["Deaths"].fillna(0).astype(int)', language="python")
            

            st.markdown("""
**2. Rename columns for clarity:**
            """)
            st.code('# Rename columns\ndf.rename(columns={"Cause of death": "Cause", "Cause of death Code": "ICD_Code"})', language="python")
            
            # Show before and after with actual column names
            cols_before = data.columns.tolist()[:5]  # Show first 5 columns
            
            rename_dict = {}
            if 'Cause of death' in data.columns:
                rename_dict['Cause of death'] = 'Cause'
            if 'Cause of death Code' in data.columns:
                rename_dict['Cause of death Code'] = 'ICD_Code'
            
            renamed = data.rename(columns=rename_dict)
            cols_after = renamed.columns.tolist()[:5]  # Show first 5 columns
            
            st.write("Original column names:")
            st.code(f"{cols_before}")
            st.write("After renaming:")
            st.code(f"{cols_after}")
            
            st.markdown("""
**3. Create simple calculated columns:**
            """)
            if all(col in data.columns for col in ['Deaths', 'Population']):
                st.code('# Create death rate column\ndf["Death_Rate"] = df["Deaths"] / df["Population"] * 100000', language="python")
                
                # Create and show the calculation with actual data
                calc_df = data.copy()
                calc_df['Deaths'] = pd.to_numeric(calc_df['Deaths'], errors='coerce')
                calc_df['Population'] = pd.to_numeric(calc_df['Population'], errors='coerce')
                calc_df['Death_Rate'] = (calc_df['Deaths'] / calc_df['Population'] * 100000).round(2)
                
                st.dataframe(calc_df[['Deaths', 'Population', 'Death_Rate']].head())
            else:
                st.code('# Create a new column\ndf["Column_C"] = df["Column_A"] + df["Column_B"]')
    
    # -------------------------------------------------------------------------
    # INTERMEDIATE LEVEL
    # -------------------------------------------------------------------------
    with tab_intermediate:
        st.markdown("### 🟡 Intermediate Level: Advanced Cleaning Techniques")
        
        with st.expander("🧮 String Methods for Cleaning"):
            st.markdown("""
### String Cleaning Methods

Text data often requires cleanup to standardize format and remove inconsistencies.

**1. Standardize case and remove whitespace:**
            """)
            st.code('# Standardize text data\ndf["State"] = df["State"].str.strip().str.lower()', language="python")
            
            # Show example with actual data if column exists
            if 'State' in data.columns:
                sample_states = data['State'].sample(5, random_state=42)
                cleaned_states = sample_states.str.strip().str.lower()
                
                st.write("Original samples:")
                st.code(f"{sample_states.tolist()}")
                st.write("After cleaning:")
                st.code(f"{cleaned_states.tolist()}")
            
            st.markdown("""
**2. Extract information from text:**
            """)
            st.code('# Extract first character of ICD codes\ndf["ICD_Category"] = df["Cause of death Code"].str[0]', language="python")
            
            # Show example with actual data if column exists
            if 'Cause of death Code' in data.columns:
                sample = data[['Cause of death Code']].head(5)
                sample['ICD_Category'] = sample['Cause of death Code'].str[0]
                
                st.dataframe(sample)
            
            st.markdown("""
**3. Replace patterns in text:**
            """)
            st.code('''# Replace text patterns
df["Cause"] = df["Cause"].str.replace(" - ", ": ")''', language="python")
            
            # Show example with actual data if column exists
            if 'Cause of death' in data.columns:
                sample_causes = data['Cause of death'].head(3)
                if sample_causes.str.contains(" - ").any():
                    replaced = sample_causes.str.replace(" - ", ": ")
                    
                    st.write("Original:")
                    st.code(f"{sample_causes.tolist()}")
                    st.write("After replacement:")
                    st.code(f"{replaced.tolist()}")
                else:
                    st.write("Example with sample data:")
                    sample_text = ["Heart disease - acute", "Cancer - lung", "Injury - fall"]
                    replaced = [s.replace(" - ", ": ") for s in sample_text]
                    
                    st.write("Original:")
                    st.code(f"{sample_text}")
                    st.write("After replacement:")
                    st.code(f"{replaced}")
        
        with st.expander("📋 Filtering and Subsetting"):
            st.markdown("""
### Advanced Filtering

Filter your data to focus on specific subsets for analysis.

**1. Filter with multiple conditions:**
            """)
            st.code('''# Filter with multiple conditions
high_death_states = df[(df["State"] == "California") & (df["Deaths"] > 100)]''', language="python")
            
            # Show example with actual data if columns exist
            if all(col in data.columns for col in ['State', 'Deaths']):
                data_copy = data.copy()
                data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                
                # Check if California exists, otherwise use first state
                if 'California' in data_copy['State'].values:
                    state_to_use = 'California'
                else:
                    state_to_use = data_copy['State'].iloc[0]
                
                filtered = data_copy[(data_copy['State'] == state_to_use) & (data_copy['Deaths'] > 100)]
                
                st.write(f"Rows where State is '{state_to_use}' AND Deaths > 100:")
                st.code(f"Number of rows: {len(filtered)}")
                if len(filtered) > 0:
                    st.dataframe(filtered.head(3))
            
            st.markdown("""
**2. Filter with OR conditions:**
            """)
            st.code('''# Filter with OR conditions
selected_states = df[(df["State"] == "Alabama") | (df["State"] == "Alaska")]''', language="python")
            
            # Show example with actual data if column exists
            if 'State' in data.columns:
                # Find two states that exist in the data
                available_states = data['State'].unique()
                if len(available_states) >= 2:
                    state1, state2 = available_states[:2]
                    
                    or_filtered = data[(data['State'] == state1) | (data['State'] == state2)]
                    
                    st.write(f"Rows where State is '{state1}' OR '{state2}':")
                    st.code(f"Number of rows: {len(or_filtered)}")
                    
                    # Count by state
                    state_counts = or_filtered['State'].value_counts()
                    st.dataframe(state_counts)
            
            st.markdown("""
**3. Filter with the `.isin()` method:**
            """)
            st.code('''# Filter with .isin()
states_of_interest = ['Alabama', 'Alaska', 'Arizona']
multi_state_data = df[df["State"].isin(states_of_interest)]''', language="python")
            
            # Show example with actual data if column exists
            if 'State' in data.columns:
                # Find states that exist in the data
                available_states = data['State'].unique()
                if len(available_states) >= 3:
                    states_to_use = available_states[:3]
                    
                    isin_filtered = data[data['State'].isin(states_to_use)]
                    
                    st.write(f"Rows where State is in {states_to_use.tolist()}:")
                    st.code(f"Number of rows: {len(isin_filtered)}")
                    
                    # Count by state
                    state_counts = isin_filtered['State'].value_counts()
                    st.dataframe(state_counts)
        
        with st.expander("📊 Grouping and Aggregation"):
            st.markdown("""
### Grouping and Aggregation

Group your data to get insights at different levels of granularity.

**1. Basic groupby operations:**
            """)
            st.code('''# Group by State and count records
df.groupby("State").size()''', language="python")
            
            # Show example with actual data if column exists
            if 'State' in data.columns:
                state_counts = data.groupby('State').size()
                st.dataframe(state_counts.head())
                
            st.markdown("""
**2. Group and calculate multiple statistics:**
            """)
            st.code('''# Group and calculate multiple statistics
df.groupby("State")["Deaths"].agg(["count", "sum", "mean", "max"])''', language="python")
            
            # Show example with actual data if columns exist
            if all(col in data.columns for col in ['State', 'Deaths']):
                data_copy = data.copy()
                data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                
                agg_stats = data_copy.groupby('State')['Deaths'].agg(['count', 'sum', 'mean', 'max'])
                st.dataframe(agg_stats.head())
            
            st.markdown("""
**3. Group by multiple columns:**
            """)
            st.code('''# Group by multiple columns
df.groupby(["State", "ICD_Category"])["Deaths"].sum()''', language="python")
            
            # Show example with actual data if possible
            if 'State' in data.columns:
                # Use ICD category if available, otherwise use another categorical column
                if 'Cause of death Code' in data.columns:
                    data_copy = data.copy()
                    data_copy['ICD_Category'] = data_copy['Cause of death Code'].str[0]
                    
                    if 'Deaths' in data.columns:
                        data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                        multi_group = data_copy.groupby(['State', 'ICD_Category'])['Deaths'].sum()
                        st.dataframe(multi_group.head(10))
                    else:
                        multi_group = data_copy.groupby(['State', 'ICD_Category']).size()
                        st.dataframe(multi_group.head(10))
                elif len(data.select_dtypes(include=['object']).columns) >= 2:
                    # Use first two object columns
                    cat_cols = data.select_dtypes(include=['object']).columns[:2]
                    multi_group = data.groupby(cat_cols.tolist()).size()
                    st.dataframe(multi_group.head(10))
    
    # -------------------------------------------------------------------------
    # ADVANCED LEVEL
    # -------------------------------------------------------------------------
    with tab_advanced:
        st.markdown("### 🔴 Advanced Level: Complex Data Transformations")
        
        with st.expander("🧩 Apply Functions to Transform Data"):
            st.markdown("""
### Using apply() for Custom Transformations

The `apply()` method lets you transform data using custom functions.

**1. Apply a function to a column:**
            """)
            st.code('''# Categorize deaths using apply()
def categorize_deaths(value):
    if pd.isna(value):
        return "Unknown"
    elif value < 10:
        return "Very Low"
    elif value < 50:
        return "Low"
    elif value < 100:
        return "Medium"
    else:
        return "High"

df["Death_Category"] = df["Deaths"].apply(categorize_deaths)''', language="python")
            
            # Show example with actual data if column exists
            if 'Deaths' in data.columns:
                data_copy = data.copy()
                data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                
                def categorize_deaths(value):
                    if pd.isna(value):
                        return "Unknown"
                    elif value < 10:
                        return "Very Low"
                    elif value < 50:
                        return "Low"
                    elif value < 100:
                        return "Medium"
                    else:
                        return "High"
                
                data_copy['Death_Category'] = data_copy['Deaths'].apply(categorize_deaths)
                
                # Show distribution of categories
                category_counts = data_copy['Death_Category'].value_counts()
                st.write("Distribution of death categories:")
                st.dataframe(category_counts)
                
            
            st.markdown("""
**2. Apply a function to rows with axis=1:**
            """)
            st.code('''# Calculate death rate using apply on rows
df["Death_Rate"] = df.apply(
    lambda row: (row["Deaths"] / row["Population"] * 100000).round(2) 
    if row["Population"] > 0 else None, 
    axis=1
)''', language="python")
            
            # Show example with actual data if columns exist
            if all(col in data.columns for col in ['Deaths', 'Population']):
                data_copy = data.copy()
                data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                data_copy['Population'] = pd.to_numeric(data_copy['Population'], errors='coerce')
                
                # Apply function to rows
                data_copy['Death_Rate'] = data_copy.apply(
                    lambda row: round(row['Deaths'] / row['Population'] * 100000, 2) 
                    if not pd.isna(row['Deaths']) and not pd.isna(row['Population']) and row['Population'] > 0 
                    else None,
                    axis=1
                )
                
                st.write("Result of apply with lambda:")
                st.dataframe(data_copy[['Deaths', 'Population', 'Death_Rate']].head())
            
            st.markdown("""
**3. Apply a function to multiple columns:**
            """)
            st.code('''# Apply the same function to multiple columns
for col in ["Deaths", "Population"]:
    df[f"{col}_log"] = df[col].apply(lambda x: np.log10(x) if x > 0 else None)''', language="python")
            
            # Show example with actual data if columns exist
            if all(col in data.columns for col in ['Deaths', 'Population']):
                import numpy as np
                
                data_copy = data.copy()
                data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                data_copy['Population'] = pd.to_numeric(data_copy['Population'], errors='coerce')
                
                # Apply log transformation
                for col in ['Deaths', 'Population']:
                    data_copy[f"{col}_log"] = data_copy[col].apply(lambda x: np.log10(x) if pd.notnull(x) and x > 0 else None)
                
                st.write("Original and log-transformed values:")
                st.dataframe(data_copy[['Deaths', 'Deaths_log', 'Population', 'Population_log']].head())
        
        with st.expander("🔄 Creating Pivot Tables"):
            st.markdown("""
### Pivot Tables for Summarizing Data

Pivot tables help reshape and summarize your data for analysis.

**1. Basic pivot table:**
            """)
            st.code('''# Create a basic pivot table
pivot = pd.pivot_table(
    df, 
    values="Deaths", 
    index="State", 
    aggfunc="sum"
)''', language="python")
            
            # Show example with actual data if columns exist
            if all(col in data.columns for col in ['State', 'Deaths']):
                data_copy = data.copy()
                data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                
                pivot = pd.pivot_table(
                    data_copy,
                    values="Deaths",
                    index="State",
                    aggfunc="sum"
                )
                
                st.write("Basic pivot table:")
                st.dataframe(pivot.head())
            
            st.markdown("""
**2. Pivot with multiple columns and values:**
            """)
            st.code('''# Create a pivot with multiple indices and columns
pivot_multi = pd.pivot_table(
    df, 
    values="Deaths", 
    index=["State"], 
    columns=["ICD_Category"],
    aggfunc="sum",
    fill_value=0
)''', language="python")
            
            # Show example with actual data if possible
            if all(col in data.columns for col in ['State', 'Deaths', 'Cause of death Code']):
                data_copy = data.copy()
                data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                data_copy['ICD_Category'] = data_copy['Cause of death Code'].str[0]
                
                # Get a subset of data for better display
                top_states = data_copy.groupby('State')['Deaths'].sum().nlargest(5).index.tolist()
                subset = data_copy[data_copy['State'].isin(top_states)]
                
                pivot_multi = pd.pivot_table(
                    subset,
                    values="Deaths",
                    index=["State"],
                    columns=["ICD_Category"],
                    aggfunc="sum",
                    fill_value=0
                )
                
                st.write("Multi-column pivot table:")
                st.dataframe(pivot_multi.head())
            
            st.markdown("""
**3. Pivot with multiple aggregation functions:**
            """)
            st.code('''# Pivot with multiple aggregation functions
pivot_aggs = pd.pivot_table(
    df, 
    values="Deaths", 
    index="State", 
    aggfunc=["sum", "mean", "count"]
)''', language="python")
            
            # Show example with actual data if columns exist
            if all(col in data.columns for col in ['State', 'Deaths']):
                data_copy = data.copy()
                data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                
                pivot_aggs = pd.pivot_table(
                    data_copy,
                    values="Deaths",
                    index="State",
                    aggfunc=["sum", "mean", "count"]
                )
                
                st.write("Pivot with multiple aggregations:")
                st.dataframe(pivot_aggs.head())
        
        with st.expander("🧬 Advanced Data Reshaping"):
            st.markdown("""
### Reshape Your Data for Different Analyses

These techniques help transform your data structure to fit specific analysis needs.

**1. Melt: Convert wide to long format**
            """)
            st.code('''# Melt example (wide to long format)
# Assuming we have mortality data by year in wide format
wide_df = pd.DataFrame({
    'State': ['Alabama', 'Alaska', 'Arizona'],
    'Deaths_2018': [100, 50, 150],
    'Deaths_2019': [110, 55, 160],
    'Deaths_2020': [130, 60, 180]
})

# Melt to long format
long_df = pd.melt(
    wide_df,
    id_vars=['State'],
    value_vars=['Deaths_2018', 'Deaths_2019', 'Deaths_2020'],
    var_name='Year',
    value_name='Deaths'
)''', language="python")
            
            # Show example with demo data
            wide_df = pd.DataFrame({
                'State': ['Alabama', 'Alaska', 'Arizona'],
                'Deaths_2018': [100, 50, 150],
                'Deaths_2019': [110, 55, 160],
                'Deaths_2020': [130, 60, 180]
            })
            
            long_df = pd.melt(
                wide_df,
                id_vars=['State'],
                value_vars=['Deaths_2018', 'Deaths_2019', 'Deaths_2020'],
                var_name='Year',
                value_name='Deaths'
            )
            
            st.write("Original wide format:")
            st.dataframe(wide_df)
            
            st.write("After melt (long format):")
            st.dataframe(long_df)
            
            st.markdown("""
**2. Stack and unstack for reshaping**
            """)
            st.code('''# Stack example
# Start with multi-level columns
df_stacked = df.set_index(['State', 'ICD_Category'])['Deaths']
# Unstack to create a pivot-like table
df_unstacked = df_stacked.unstack()''', language="python")
            
            # Show example with actual data if possible
            if all(col in data.columns for col in ['State', 'Deaths', 'Cause of death Code']):
                data_copy = data.copy()
                data_copy['Deaths'] = pd.to_numeric(data_copy['Deaths'], errors='coerce')
                data_copy['ICD_Category'] = data_copy['Cause of death Code'].str[0]
                
                # Get a subset for better display
                subset = data_copy.head(20)  # Just use a small sample
                
                # Stack/unstack example
                try:
                    df_stacked = subset.set_index(['State', 'ICD_Category'])['Deaths']
                    df_unstacked = df_stacked.unstack()
                    
                    st.write("After stacking and unstacking:")
                    st.dataframe(df_unstacked.head())
                except Exception as e:
                    st.code(f"Error demonstrating stack/unstack: {e}")
                    
                    # Show simplified example
                    st.write("Simplified example:")
                    simple_df = pd.DataFrame({
                        'State': ['AL', 'AL', 'FL', 'FL'],
                        'Category': ['A', 'B', 'A', 'B'],
                        'Value': [10, 20, 30, 40]
                    })
                    
                    st.dataframe(simple_df)
                    
                    stacked = simple_df.set_index(['State', 'Category'])['Value']
                    unstacked = stacked.unstack()
                    
                    st.write("After unstacking:")
                    st.dataframe(unstacked)
                    
            st.markdown("""
**3. Merge datasets**
            """)
            st.code('''# Merge example
# Assume we have state population data
state_pop = pd.DataFrame({
    'State': ['Alabama', 'Alaska', 'Arizona'],
    'Population': [5000000, 700000, 7300000]
})

# Merge with our mortality data
merged_df = pd.merge(
    df, 
    state_pop,
    on='State',
    how='left'
)''', language="python")
            
            # Show example with demo data
            if 'State' in data.columns:
                # Create sample state population data
                available_states = data['State'].unique()
                if len(available_states) >= 3:
                    sample_states = available_states[:3]
                    
                    state_pop = pd.DataFrame({
                        'State': sample_states,
                        'Total_Population': [5000000, 700000, 7300000]
                    })
                    
                    st.write("State population data:")
                    st.dataframe(state_pop)
                    
                    # Create small subset of main data
                    subset = data[data['State'].isin(sample_states)].head(6)
                    
                    # Do the merge
                    merged_df = pd.merge(
                        subset,
                        state_pop,
                        on='State',
                        how='left'
                    )
                    
                    st.write("After merging:")
                    st.dataframe(merged_df)
                    
                    
             
elif st.session_state.page == "step5_eda":
    st.header("📊 Step 5 – Exploratory Data Analysis (EDA)")
    if st.button("← Back to Knowledge Graph", key="back_btn_step5"):
        st.session_state.page = "graph"
        st.session_state.reset_guidance = True
        st.rerun()
    
    st.subheader("Choose Your Level")
    tab1, tab2, tab3 = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"])
    
    # 🟢 Beginner: Basic EDA and Filtering Techniques
    with tab1:
        st.markdown("### 🟢 Beginner: Basic EDA and Filtering")
        
        with st.expander("1. Accessing Columns"):
            st.markdown("""
### Accessing Columns in a DataFrame

**Objective**:  
Learn how to select specific columns from a DataFrame.

**Purpose**:  
Accessing individual columns allows you to view, modify, or perform operations on specific data.

**Key Points**:
- Use `df["Column_Name"]` to select a single column (returns a Series)
- Use `df[["Column_Name1", "Column_Name2"]]` to select multiple columns (returns a DataFrame)
- Single brackets return a Series, double brackets return a DataFrame

**Example 1**: Select a single column
            """)
            st.code('''# Access the 'State' column
state_column = data["State"]
print(type(state_column))  # This will show pandas.core.series.Series
print(state_column.head())''', language="python")
            
            st.write("**Output:**")
            import io
            buffer = io.StringIO()
            print(type(data["State"]), file=buffer)
            print(data["State"].head(), file=buffer)
            st.code(buffer.getvalue())
            
            st.markdown("""
**Example 2**: Select multiple columns
            """)
            st.code('''# Access multiple columns
selected_columns = data[["State", "Deaths", "Population"]]
print(type(selected_columns))  # This will show pandas.core.frame.DataFrame
print(selected_columns.head())''', language="python")
            
            st.write("**Output:**")
            # Get the actual data outputs
            buffer = io.StringIO()
            print(type(data[["State", "Deaths", "Population"]]), file=buffer)
            st.code(buffer.getvalue())
            st.dataframe(data[["State", "Deaths", "Population"]].head())
        
        with st.expander("2. Simple Filtering with Conditions"):
            st.markdown("""
### Simple Filtering with Conditions

**Objective**:  
Learn how to filter rows in a DataFrame based on a condition.

**Purpose**:  
Filtering allows you to focus on specific subsets of your data that meet certain criteria.

**Key Points**:
- Use boolean expressions to create masks
- Apply these masks to the DataFrame using square brackets
- Conditions can involve comparison operators like `>`, `<`, `==`, `!=`, `>=`, `<=`

**Example 1**: Filter rows based on a numeric threshold
            """)
            st.code('''# Filter rows where 'Deaths' is greater than 100
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
high_deaths = data_copy[data_copy["Deaths"] > 100]
print(f"Number of rows with Deaths > 100: {len(high_deaths)}")
print(high_deaths.head())''', language="python")
            
            st.write("**Output:**")
            # Get the actual output
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            high_deaths = data_copy[data_copy["Deaths"] > 100]
            st.code(f"Number of rows with Deaths > 100: {len(high_deaths)}")
            st.dataframe(high_deaths.head())
            
            st.markdown("""
**Example 2**: Filter rows based on a text condition
            """)
            st.code('''# Filter rows where State is 'Alabama'
alabama_data = data[data["State"] == "Alabama"]
print(f"Number of rows for Alabama: {len(alabama_data)}")
print(alabama_data.head(3))''', language="python")
            
            st.write("**Output:**")
            # Get the actual output for Alabama filter
            alabama_data = data[data["State"] == "Alabama"]
            st.code(f"Number of rows for Alabama: {len(alabama_data)}")
            st.dataframe(alabama_data.head(3))
        
        with st.expander("3. Basic Distribution Analysis"):
            st.markdown("""
### Basic Distribution Analysis

**Objective**:  
Visualize and understand the distribution of a variable in your dataset.

**Purpose**:  
Understanding the distribution helps identify patterns, outliers, and the general shape of your data.

**Example 1**: Histogram of Deaths
            """)
            st.code('''# Create a histogram of the 'Deaths' column
import matplotlib.pyplot as plt
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

plt.figure(figsize=(10, 6))
plt.hist(data_copy["Deaths"].dropna(), bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Deaths")
plt.xlabel("Number of Deaths")
plt.ylabel("Frequency")
plt.show()''', language="python")
            
            # Create an actual histogram using the real data
            import matplotlib.pyplot as plt
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data_copy["Deaths"].dropna(), bins=20, color="skyblue", edgecolor="black")
            ax.set_title("Distribution of Deaths")
            ax.set_xlabel("Number of Deaths")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
            st.markdown("""
**Example 2**: Boxplot for outlier detection
            """)
            st.code('''# Create a boxplot to identify outliers
plt.figure(figsize=(10, 6))
plt.boxplot(data_copy["Deaths"].dropna())
plt.title("Boxplot of Deaths")
plt.ylabel("Number of Deaths")
plt.show()''', language="python")
            
            # Create an actual boxplot using the real data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(data_copy["Deaths"].dropna())
            ax.set_title("Boxplot of Deaths")
            ax.set_ylabel("Number of Deaths")
            st.pyplot(fig)
    
    # 🟡 Intermediate: More Advanced Filtering and EDA
    with tab2:
        st.markdown("### 🟡 Intermediate: Advanced Filtering and EDA")
        
        with st.expander("1. Filtering with Multiple Conditions"):
            st.markdown("""
### Filtering with Multiple Conditions

**Objective**:  
Learn how to filter data based on multiple criteria simultaneously.

**Purpose**:  
Combining conditions allows for more precise data selection and more complex analyses.

**Key Points**:
- Use `&` for AND operations (both conditions must be true)
- Use `|` for OR operations (at least one condition must be true)
- Use parentheses to group conditions for clarity and proper precedence

**Example 1**: Filter with AND condition
            """)
            st.code('''# Filter rows where State is 'Alabama' AND Deaths > 100
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
alabama_high_deaths = data_copy[(data_copy["State"] == "Alabama") & (data_copy["Deaths"] > 100)]
print(f"Number of rows: {len(alabama_high_deaths)}")
print(alabama_high_deaths.head(3))''', language="python")
            
            st.write("**Output:**")
            # Get the actual output
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            alabama_high_deaths = data_copy[(data_copy["State"] == "Alabama") & (data_copy["Deaths"] > 100)]
            st.code(f"Number of rows: {len(alabama_high_deaths)}")
            st.dataframe(alabama_high_deaths.head(3))
            
            st.markdown("""
**Example 2**: Filter with OR condition
            """)
            st.code('''# Filter rows where State is either 'Alabama' OR 'Alaska'
alabama_or_alaska = data[(data["State"] == "Alabama") | (data["State"] == "Alaska")]
print(f"Number of rows: {len(alabama_or_alaska)}")
# Count rows for each state
print(alabama_or_alaska["State"].value_counts())''', language="python")
            
            st.write("**Output:**")
            # Get the actual output
            alabama_or_alaska = data[(data["State"] == "Alabama") | (data["State"] == "Alaska")]
            buffer = io.StringIO()
            print(f"Number of rows: {len(alabama_or_alaska)}", file=buffer)
            print(alabama_or_alaska["State"].value_counts(), file=buffer)
            st.code(buffer.getvalue())
            
            st.markdown("""
**Example 3**: Complex filtering with multiple conditions
            """)
            
            # First check if Florida exists in the data
            florida_exists = "Florida" in data["State"].values
            second_state = "Florida" if florida_exists else data["State"].unique()[1]
            
            st.code(f'''# Filter for states in a specific region with high death counts
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
data_copy["Crude Rate"] = data_copy["Crude Rate"].astype(str)

selected_data = data_copy[
    ((data_copy["State"] == "Alabama") | (data_copy["State"] == "{second_state}")) & 
    (data_copy["Deaths"] > 100) & 
    (data_copy["Crude Rate"] != "Unreliable")
]
print(f"Number of rows: {{len(selected_data)}}")
print(selected_data.head(3))''', language="python")
            
            st.write("**Output:**")
            # Get the actual output
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            data_copy["Crude Rate"] = data_copy["Crude Rate"].astype(str) 
            
            selected_data = data_copy[
                ((data_copy["State"] == "Alabama") | (data_copy["State"] == second_state)) & 
                (data_copy["Deaths"] > 100) & 
                (data_copy["Crude Rate"] != "Unreliable")
            ]
            st.code(f"Number of rows: {len(selected_data)}")
            st.dataframe(selected_data.head(3))
        
        with st.expander("2. Filtering with the .isin() Method"):
            st.markdown("""
### Filtering with the .isin() Method

**Objective**:  
Learn how to filter rows based on membership in a list of values.

**Purpose**:  
The `.isin()` method provides a concise way to filter rows where a column's value matches any item in a specified list.

**Example 1**: Filter states using a list of values
            """)
            
            # Get a list of states that actually exist in the data
            available_states = data["State"].unique()
            selected_states = available_states[:min(4, len(available_states))]
            
            st.code(f'''# Create a list of states to filter
selected_states = {list(selected_states)}

# Filter rows where the State is in the list
multi_state_data = data[data["State"].isin(selected_states)]
print(f"Number of rows: {{len(multi_state_data)}}")
print(multi_state_data["State"].value_counts())''', language="python")
            
            st.write("**Output:**")
            # Get the actual output
            multi_state_data = data[data["State"].isin(selected_states)]
            buffer = io.StringIO()
            print(f"Number of rows: {len(multi_state_data)}", file=buffer)
            print(multi_state_data["State"].value_counts(), file=buffer)
            st.code(buffer.getvalue())
            
            st.markdown("""
**Example 2**: Exclude values using ~ (NOT)
            """)
            st.code(f'''# Filter rows where the State is NOT in the list (using ~ operator)
other_states = data[~data["State"].isin({list(selected_states)})]
print(f"Number of rows: {{len(other_states)}}")
print(f"Number of unique remaining states: {{other_states['State'].nunique()}}")''', language="python")
            
            st.write("**Output:**")
            # Get the actual output
            other_states = data[~data["State"].isin(selected_states)]
            buffer = io.StringIO()
            print(f"Number of rows: {len(other_states)}", file=buffer)
            print(f"Number of unique remaining states: {other_states['State'].nunique()}", file=buffer)
            st.code(buffer.getvalue())
        
        with st.expander("3. Correlation Analysis"):
            st.markdown("""
### Correlation Analysis

**Objective**:  
Examine relationships between numerical variables in your dataset.

**Purpose**:  
Correlation analysis helps identify how variables relate to each other, which can reveal potential patterns or relationships.

**Example 1**: Calculate correlation matrix
            """)
            st.code('''# Calculate the correlation between numeric columns
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

corr_matrix = data_copy[["Deaths", "Population"]].corr()
print(corr_matrix)''', language="python")
            
            st.write("**Output:**")
            # Get the actual output
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
            
            try:
                corr_matrix = data_copy[["Deaths", "Population"]].corr()
                buffer = io.StringIO()
                print(corr_matrix, file=buffer)
                st.code(buffer.getvalue())
            except Exception as e:
                st.error(f"Error calculating correlation: {e}")
                st.code("Unable to calculate correlation with this dataset. This might be due to missing values or constant values.")
            
            st.markdown("""
**Example 2**: Visualize correlation using a scatter plot
            """)
            st.code('''# Create a scatter plot of Deaths vs Population
import matplotlib.pyplot as plt
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

plt.figure(figsize=(10, 6))
plt.scatter(data_copy["Population"], data_copy["Deaths"], alpha=0.5)
plt.title("Relationship between Population and Deaths")
plt.xlabel("Population")
plt.ylabel("Deaths")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()''', language="python")
            
            # Create an actual scatter plot using the real data
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(data_copy["Population"], data_copy["Deaths"], alpha=0.5)
                ax.set_title("Relationship between Population and Deaths")
                ax.set_xlabel("Population")
                ax.set_ylabel("Deaths")
                ax.grid(True, linestyle="--", alpha=0.7)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating scatter plot: {e}")
                st.write("Unable to create scatter plot with this dataset. Try cleaning the data first.")
    
    # 🔴 Advanced: Complex Filtering and Advanced EDA
    with tab3:
        st.markdown("### 🔴 Advanced: Complex Filtering and Advanced EDA")
        
        with st.expander("1. Filtering with String Methods"):
            st.markdown("""
### Filtering with String Methods

**Objective**:  
Learn how to filter text data based on string patterns or contents.

**Purpose**:  
String methods allow for more complex text matching, enabling you to find data that contains, starts with, or matches specific patterns.

**Example 1**: Filter using string.contains()
            """)
            
            # Check if "Cause of death" column exists
            if "Cause of death" in data.columns:
                st.code('''# Filter causes of death containing the word "Malignant"
cancer_data = data[data["Cause of death"].str.contains("Malignant", case=False, na=False)]
print(f"Number of rows with 'Malignant' in the cause: {len(cancer_data)}")
print(cancer_data[["Cause of death", "Deaths"]].head(3))''', language="python")
                
                st.write("**Output:**")
                # Get the actual output
                cancer_data = data[data["Cause of death"].str.contains("Malignant", case=False, na=False)]
                st.code(f"Number of rows with 'Malignant' in the cause: {len(cancer_data)}")
                st.dataframe(cancer_data[["Cause of death", "Deaths"]].head(3))
                
                st.markdown("""
**Example 2**: Filter using string.startswith()
                """)
                st.code('''# Filter causes of death that start with 'Heart'
heart_data = data[data["Cause of death"].str.startswith("Heart", na=False)]
print(f"Number of rows starting with 'Heart': {len(heart_data)}")
print(heart_data[["Cause of death", "Deaths"]].head(3))''', language="python")
                
                st.write("**Output:**")
                # Get the actual output
                heart_data = data[data["Cause of death"].str.startswith("Heart", na=False)]
                st.code(f"Number of rows starting with 'Heart': {len(heart_data)}")
                st.dataframe(heart_data[["Cause of death", "Deaths"]].head(3))
                
                st.markdown("""
**Example 3**: Filter using regular expressions
                """)
                st.code('''# Filter using a regular expression to find all respiratory conditions
import re
respiratory = data[data["Cause of death"].str.contains(r"lung|bronch|respir|pneumo", 
                                                case=False, regex=True, na=False)]
print(f"Number of respiratory-related rows: {len(respiratory)}")
print(respiratory["Cause of death"].value_counts().head(3))''', language="python")
                
                st.write("**Output:**")
                # Get the actual output
                respiratory = data[data["Cause of death"].str.contains(r"lung|bronch|respir|pneumo", 
                                                            case=False, regex=True, na=False)]
                buffer = io.StringIO()
                print(f"Number of respiratory-related rows: {len(respiratory)}", file=buffer)
                print(respiratory["Cause of death"].value_counts().head(3), file=buffer)
                st.code(buffer.getvalue())
            else:
                # If "Cause of death" doesn't exist, use another text column
                text_columns = data.select_dtypes(include=['object']).columns
                if len(text_columns) > 0:
                    text_col = text_columns[0]
                    search_term = data[text_col].iloc[0][:5] if len(data) > 0 else "test"
                    
                    st.code(f'''# Filter rows where {text_col} contains "{search_term}"
filtered_data = data[data["{text_col}"].str.contains("{search_term}", case=False, na=False)]
print(f"Number of rows with '{search_term}' in {text_col}: {{len(filtered_data)}}")
print(filtered_data.head(3))''', language="python")
                    
                    st.write("**Output:**")
                    # Get the actual output
                    filtered_data = data[data[text_col].str.contains(search_term, case=False, na=False)]
                    st.code(f"Number of rows with '{search_term}' in {text_col}: {len(filtered_data)}")
                    st.dataframe(filtered_data.head(3))
                else:
                    st.warning("No text columns found in the dataset to demonstrate string filtering methods.")
        
        with st.expander("2. Advanced Multivariate Analysis"):
            st.markdown("""
### Advanced Multivariate Analysis

**Objective**:  
Examine complex relationships between multiple variables simultaneously.

**Purpose**:  
Multivariate analysis reveals insights that might not be apparent when examining variables in isolation.

**Example 1**: Create a pivot table
            """)
            
            # Check if "Cause of death Code" column exists for ICD chapter
            if "Cause of death Code" in data.columns:
                st.code('''# Create a pivot table of average deaths by state and ICD chapter
import pandas as pd

# Create an ICD chapter column if it doesn't exist
data_copy = data.copy()
# Create a simplified mapping function
def get_icd_chapter(code):
    if pd.isna(code):
        return "Unknown"
    first_char = str(code)[0]
    if first_char in "AB":
        return "Infectious Diseases"
    elif first_char == "C":
        return "Neoplasms"
    elif first_char == "I":
        return "Circulatory System"
    elif first_char == "J":
        return "Respiratory System"
    else:
        return "Other"

data_copy["ICD Chapter"] = data_copy["Cause of death Code"].apply(get_icd_chapter)
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Create the pivot table
pivot = pd.pivot_table(data_copy, 
                       values="Deaths",
                       index="State", 
                       columns="ICD Chapter", 
                       aggfunc="mean",
                       fill_value=0)

print(pivot.head())''', language="python")
                
                st.write("**Output:**")
                # Get the actual output
                data_copy = data.copy()
                def get_icd_chapter(code):
                    if pd.isna(code):
                        return "Unknown"
                    first_char = str(code)[0]
                    if first_char in "AB":
                        return "Infectious Diseases"
                    elif first_char == "C":
                        return "Neoplasms"
                    elif first_char == "I":
                        return "Circulatory System"
                    elif first_char == "J":
                        return "Respiratory System"
                    else:
                        return "Other"
                
                data_copy["ICD Chapter"] = data_copy["Cause of death Code"].apply(get_icd_chapter)
                data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
                
                pivot = pd.pivot_table(data_copy, 
                                       values="Deaths",
                                       index="State", 
                                       columns="ICD Chapter", 
                                       aggfunc="mean",
                                       fill_value=0)
                
                st.dataframe(pivot.head())
                
                st.markdown("""
**Example 2**: Create a heatmap of the pivot table
                """)
                st.code('''# Visualize the pivot table as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(pivot, cmap="YlOrRd", annot=False)
plt.title("Average Deaths by State and ICD Chapter")
plt.tight_layout()
plt.show()''', language="python")
                
                # Create an actual heatmap using the real data
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(pivot, cmap="YlOrRd", annot=False, ax=ax)
                ax.set_title("Average Deaths by State and ICD Chapter")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                # Create a simple pivot table with available columns
                st.code('''# Create a pivot table with available columns
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Create the pivot table using State as index
pivot = pd.pivot_table(data_copy, 
                       values="Deaths",
                       index="State", 
                       aggfunc=["mean", "sum", "count"],
                       fill_value=0)

print(pivot.head())''', language="python")
                
                st.write("**Output:**")
                # Get the actual output
                data_copy = data.copy()
                data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
                
                pivot = pd.pivot_table(data_copy, 
                                      values="Deaths",
                                      index="State", 
                                      aggfunc=["mean", "sum", "count"],
                                      fill_value=0)

                st.dataframe(pivot.head())
                
                st.markdown("""
**Example 2**: Create a heatmap of simple statistics
                """)
                
                st.code('''# Create a heatmap of summary statistics
import seaborn as sns
import matplotlib.pyplot as plt

# Convert the pivot table to a more visualization-friendly format
pivot_display = pivot.head(10)  # Just show 10 states for better visibility
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_display, cmap="YlOrRd", annot=True, fmt=".1f")
plt.title("Death Statistics by State")
plt.tight_layout()
plt.show()''', language="python")
                
                # Create the actual heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                pivot_display = pivot.head(10)  # Just show 10 states for better visibility
                sns.heatmap(pivot_display, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax)
                ax.set_title("Death Statistics by State")
                plt.tight_layout()
                st.pyplot(fig)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
elif st.session_state.page == "step6_visualization":
    st.header("🎨 Step 6 – Visualization")
    if st.button("← Back to Knowledge Graph", key="back_btn_step6"):
        st.session_state.page = "graph"
        st.session_state.reset_guidance = True
        st.rerun()
    st.subheader("Choose Your Level")
    tab1, tab2, tab3 = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"])
    
    # 🟢 Beginner: Basic Visualizations
    with tab1:
        st.markdown("### 🟢 Beginner: Basic Visualizations")
        with st.expander("1. Line Chart"):
            st.markdown("""
### Line Chart

**Objective:**  
Create a simple line chart to visualize data trends.

**Key Points:**
- Line charts are best for showing trends over time or ordered categories
- They work well when you have a clear progression in your data
- In our case, we'll use States as categories for demonstration

**Example:**  
Plot a line chart of total Deaths by State (using State as the x-axis):
            """)
            st.code(
'''# Convert Deaths to numeric and sort by State for better visualization
import matplotlib.pyplot as plt

data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Group by State and sum Deaths
state_deaths = data_copy.groupby("State")["Deaths"].sum().reset_index()
state_deaths = state_deaths.sort_values("State")

# Create the line chart
plt.figure(figsize=(12, 6))
plt.plot(state_deaths["State"], state_deaths["Deaths"], marker="o", linestyle="-")
plt.xticks(rotation=90)
plt.title("Total Deaths by State")
plt.xlabel("State")
plt.ylabel("Total Deaths")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            import matplotlib.pyplot as plt
            
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Group by State and sum Deaths
            state_deaths = data_copy.groupby("State")["Deaths"].sum().reset_index()
            state_deaths = state_deaths.sort_values("State")
            
            # Create the line chart
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(state_deaths["State"], state_deaths["Deaths"], marker="o", linestyle="-")
            plt.xticks(rotation=90)
            ax.set_title("Total Deaths by State")
            ax.set_xlabel("State")
            ax.set_ylabel("Total Deaths")
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
**Insights:**
- Line charts help you quickly identify states with higher or lower death counts
- The connecting lines help your eye follow the pattern across states
- When data isn't inherently sequential (like states), bar charts might be more appropriate
            """)
            
        with st.expander("2. Bar Chart"):
            st.markdown("""
### Bar Chart

**Objective:**  
Create a bar chart to compare categorical data.

**Key Points:**
- Bar charts are ideal for comparing values across categories
- They work well for displaying summary statistics by group
- Horizontal bar charts can be better for many categories or long labels

**Example:**  
Display a bar chart of the top states by total Deaths:
            """)
            st.code(
'''# Convert Deaths to numeric and calculate totals by state
import matplotlib.pyplot as plt

data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Group by State, sum Deaths, and get top 10
top_states = data_copy.groupby("State")["Deaths"].sum().nlargest(10)

# Create the bar chart
plt.figure(figsize=(10, 6))
top_states.plot(kind="bar", color="skyblue")
plt.title("Top 10 States by Total Deaths")
plt.xlabel("State")
plt.ylabel("Total Deaths")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Group by State, sum Deaths, and get top 10
            top_states = data_copy.groupby("State")["Deaths"].sum().nlargest(10)
            
            # Create the bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            top_states.plot(kind="bar", color="skyblue", ax=ax)
            ax.set_title("Top 10 States by Total Deaths")
            ax.set_xlabel("State")
            ax.set_ylabel("Total Deaths")
            plt.xticks(rotation=45)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
**Insights:**
- Bar charts clearly show which states have the highest death counts
- The ordering from highest to lowest makes it easy to identify rankings
- For comparing many categories, consider a horizontal bar chart instead
            """)
            
        with st.expander("3. Pie Chart"):
            st.markdown("""
### Pie Chart

**Objective:**  
Create a pie chart to show proportions of a whole.

**Key Points:**
- Pie charts show how individual parts relate to the whole
- Best used when you have a small number of categories (generally 5-7 max)
- Always sum to 100%, making percentage comparisons intuitive

**Example:**  
Create a pie chart showing the distribution of deaths among the top 5 states:
            """)
            st.code(
'''# Convert Deaths to numeric and calculate totals by state
import matplotlib.pyplot as plt

data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Group by State, sum Deaths, and get top 5
top_5_states = data_copy.groupby("State")["Deaths"].sum().nlargest(5)
other_states = data_copy.groupby("State")["Deaths"].sum().nsmallest(len(data_copy["State"].unique()) - 5).sum()

# Combine top 5 with 'Other'
pie_data = top_5_states.copy()
pie_data["Other States"] = other_states

# Create the pie chart
plt.figure(figsize=(10, 8))
plt.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90, shadow=True)
plt.title("Distribution of Deaths: Top 5 States vs Others")
plt.axis("equal")  # Equal aspect ratio ensures the pie is circular
plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Group by State, sum Deaths, and get top 5
            top_5_states = data_copy.groupby("State")["Deaths"].sum().nlargest(5)
            other_states = data_copy.groupby("State")["Deaths"].sum().nsmallest(len(data_copy["State"].unique()) - 5).sum()
            
            # Combine top 5 with 'Other'
            pie_data = top_5_states.copy()
            pie_data["Other States"] = other_states
            
            # Create the pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90, shadow=True)
            ax.set_title("Distribution of Deaths: Top 5 States vs Others")
            ax.axis("equal")  # Equal aspect ratio ensures the pie is circular
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
**Insights:**
- Pie charts show which states account for the largest proportions of deaths
- Combining smaller categories into "Other" helps maintain chart readability
- The percentage labels make it easy to understand the relative size of each slice

**When to avoid pie charts:**
- When you have too many categories (becomes cluttered)
- When differences between slices are small (hard to interpret)
- When precise comparisons are important (bar charts are better)
            """)
            
        with st.expander("4. Histogram"):
            st.markdown("""
### Histogram

**Objective:**  
Create a histogram to visualize the distribution of a numerical variable.

**Key Points:**
- Histograms show the frequency distribution of a continuous variable
- They help identify the shape, central tendency, and spread of your data
- The number of bins affects how detailed the distribution appears

**Example:**  
Create a histogram of the Deaths column to see its distribution:
            """)
            st.code(
'''# Convert Deaths to numeric and create a histogram
import matplotlib.pyplot as plt

data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(data_copy["Deaths"].dropna(), bins=30, color="skyblue", edgecolor="black")
plt.title("Distribution of Death Counts")
plt.xlabel("Number of Deaths")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Create the histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data_copy["Deaths"].dropna(), bins=30, color="skyblue", edgecolor="black")
            ax.set_title("Distribution of Death Counts")
            ax.set_xlabel("Number of Deaths")
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
**Insights:**
- The histogram reveals the overall distribution pattern of death counts
- Most entries fall on the lower end of the scale, creating a right-skewed distribution
- A few very high values (outliers) stretch the distribution to the right
- This kind of distribution is common in real-world data and often benefits from log transformation
            """)
    
    # 🟡 Intermediate: Combined Visualizations and Log Scales
    with tab2:
        st.markdown("### 🟡 Intermediate: Combined Visualizations and Log Scales")
        
        with st.expander("1. Multiple Charts Side by Side"):
            st.markdown("""
### Multiple Charts Side by Side

**Objective:**  
Create multiple charts in a single figure to compare different visualizations.

**Key Points:**
- Multiple charts allow direct comparison between different visualizations
- Use `plt.subplots()` to create a grid of axes for multiple charts
- Make sure to use `plt.tight_layout()` to prevent overlapping

**Example:**  
Create a line chart and bar chart side by side for comparison:
            """)
            st.code(
'''# Create multiple charts to compare different views of the same data
import matplotlib.pyplot as plt
import pandas as pd

data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Prepare the data for plotting
state_deaths = data_copy.groupby("State")["Deaths"].sum().sort_values(ascending=False)
top_states = state_deaths.head(5)

# Create subplots - 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Line chart of all states
state_deaths.sort_index().plot(ax=ax1, kind="line", marker="o", color="blue")
ax1.set_title("Line Chart: Deaths by State (Alphabetical)")
ax1.set_xlabel("State")
ax1.set_ylabel("Total Deaths")
ax1.tick_params(axis="x", rotation=90)
ax1.grid(True, linestyle="--", alpha=0.7)

# Plot1 2: Bar chart of top 5 states
top_states.plot(ax=ax2, kind="bar", color="green")
ax2.set_title("Bar Chart: Top 5 States by Deaths")
ax2.set_xlabel("State")
ax2.set_ylabel("Total Deaths")
ax2.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Prepare the data for plotting
            state_deaths = data_copy.groupby("State")["Deaths"].sum().sort_values(ascending=False)
            top_states = state_deaths.head(5)
            
            # Create subplots - 1 row, 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Line chart of all states
            state_deaths.sort_index().plot(ax=ax1, kind="line", marker="o", color="blue")
            ax1.set_title("Line Chart: Deaths by State (Alphabetical)")
            ax1.set_xlabel("State")
            ax1.set_ylabel("Total Deaths")
            ax1.tick_params(axis="x", rotation=90)
            ax1.grid(True, linestyle="--", alpha=0.7)
            
            # Plot 2: Bar chart of top 5 states
            top_states.plot(ax=ax2, kind="bar", color="green")
            ax2.set_title("Bar Chart: Top 5 States by Deaths")
            ax2.set_xlabel("State")
            ax2.set_ylabel("Total Deaths")
            ax2.grid(axis="y", linestyle="--", alpha=0.7)
            
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
**Insights:**
- Multiple charts allow you to present different perspectives of your data simultaneously
- The line chart shows all states but makes it hard to compare values precisely
- The bar chart clearly shows the top 5 states but doesn't include all data
- Together, they provide a more comprehensive view of the dataset
            """)
            
        with st.expander("2. Understanding Log Scales"):
            st.markdown("""
### Understanding Log Scales

**Objective:**  
Learn when and how to use logarithmic scales in data visualization.

**What are logarithmic scales?**  
Log scales transform data using logarithms (typically base 10), which compress large ranges of values to make patterns more visible across different orders of magnitude.

**When to use log scales:**
- When your data spans several orders of magnitude
- When you want to visualize percent changes or multiplicative factors
- When your data follows an exponential or power law distribution

**Example:**  
Compare linear and logarithmic scales for visualizing state population vs. deaths:
            """)
            st.code(
'''# Compare linear and logarithmic scales
import matplotlib.pyplot as plt
import numpy as np

data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

# Create a summary dataframe with state populations and death counts
state_summary = data_copy.groupby("State").agg({
    "Deaths": "sum",
    "Population": "mean"
}).reset_index()

# Create subplots - 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Linear scale scatter plot
ax1.scatter(state_summary["Population"], state_summary["Deaths"], alpha=0.7, s=80)
ax1.set_title("Population vs. Deaths (Linear Scale)")
ax1.set_xlabel("Population")
ax1.set_ylabel("Total Deaths")
ax1.grid(True, linestyle="--", alpha=0.7)

# Plot 2: Log scale scatter plot
ax2.scatter(state_summary["Population"], state_summary["Deaths"], alpha=0.7, s=80, color="red")
ax2.set_title("Population vs. Deaths (Log Scale)")
ax2.set_xlabel("Population (Log Scale)")
ax2.set_ylabel("Total Deaths (Log Scale)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.grid(True, linestyle="--", alpha=0.7)

# Add a best fit line to the log plot to show the relationship
x = state_summary["Population"]
y = state_summary["Deaths"]
coefficients = np.polyfit(np.log10(x), np.log10(y), 1)
polynomial = np.poly1d(coefficients)
x_log = np.geomspace(x.min(), x.max(), 100)
y_log = 10**(polynomial(np.log10(x_log)))
ax2.plot(x_log, y_log, color="black", linestyle="--")

# Add annotations explaining the slope
slope = coefficients[0]
ax2.annotate(f"Slope: {slope:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            import numpy as np
            
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
            
            # Create a summary dataframe with state populations and death counts
            state_summary = data_copy.groupby("State").agg({
                "Deaths": "sum",
                "Population": "mean"
            }).reset_index()
            
            # Create subplots - 1 row, 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Plot 1: Linear scale scatter plot
            ax1.scatter(state_summary["Population"], state_summary["Deaths"], alpha=0.7, s=80)
            ax1.set_title("Population vs. Deaths (Linear Scale)")
            ax1.set_xlabel("Population")
            ax1.set_ylabel("Total Deaths")
            ax1.grid(True, linestyle="--", alpha=0.7)
            
            # Plot 2: Log scale scatter plot
            ax2.scatter(state_summary["Population"], state_summary["Deaths"], alpha=0.7, s=80, color="red")
            ax2.set_title("Population vs. Deaths (Log Scale)")
            ax2.set_xlabel("Population (Log Scale)")
            ax2.set_ylabel("Total Deaths (Log Scale)")
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.grid(True, linestyle="--", alpha=0.7)
            
            # Add a best fit line to the log plot to show the relationship
            x = state_summary["Population"]
            y = state_summary["Deaths"]
            # Filter out any zeros or negative values
            mask = (x > 0) & (y > 0)
            x = x[mask]
            y = y[mask]
            if len(x) > 1 and len(y) > 1:  # Make sure we have enough data points
                coefficients = np.polyfit(np.log10(x), np.log10(y), 1)
                polynomial = np.poly1d(coefficients)
                x_log = np.geomspace(x.min(), x.max(), 100)
                y_log = 10**(polynomial(np.log10(x_log)))
                ax2.plot(x_log, y_log, color="black", linestyle="--")
                
                # Add annotations explaining the slope
                slope = coefficients[0]
                ax2.annotate(f"Slope: {slope:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
**The Power of Log Scales:**

1. **Revealing Patterns**:
   - In the linear scale (left), larger states dominate the visualization
   - In the log scale (right), the relationship between population and deaths becomes clearer
   - The best-fit line in the log-log plot reveals the power-law relationship
   
2. **Understanding the Best-Fit Line**:
   - The slope of the line in a log-log plot tells us about the relationship
   - A slope of 1.0 would mean deaths scale directly proportionally with population
   - A slope < 1.0 means deaths increase more slowly than population
   - A slope > 1.0 means deaths increase faster than population

3. **Real-World Applications**:
   - Many biological, social, and physical phenomena follow power laws visible on log scales
   - Population scaling effects (like we see here) are common in epidemiology
   - Log scales make it easier to see relationships across different orders of magnitude
            """)
            
        with st.expander("3. Seaborn Statistical Visualizations"):
            st.markdown("""
### Seaborn Statistical Visualizations

**Objective:**  
Create more sophisticated statistical visualizations using the Seaborn library.

**Key Points:**
- Seaborn builds on Matplotlib to provide more attractive and informative statistical graphics
- It integrates well with pandas DataFrames
- It offers built-in themes and color palettes for professional-looking visualizations

**Example:**  
Create a boxplot to compare death distributions across states:
            """)
            st.code(
'''# Create statistical visualizations with Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Set the Seaborn style
sns.set_style("whitegrid")

data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Get the top 10 states by median death count
state_median_deaths = data_copy.groupby("State")["Deaths"].median().sort_values(ascending=False)
top_10_states = state_median_deaths.head(10).index.tolist()

# Filter for just those states
plot_data = data_copy[data_copy["State"].isin(top_10_states)]

# Create the boxplot
plt.figure(figsize=(14, 8))
ax = sns.boxplot(x="State", y="Deaths", data=plot_data, palette="viridis")
ax.set_title("Distribution of Death Counts Across Top 10 States (by Median)", fontsize=14)
ax.set_xlabel("State", fontsize=12)
ax.set_ylabel("Deaths", fontsize=12)
ax.set_yscale("log")  # Using log scale to better visualize the distribution
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            import seaborn as sns
            
            # Set the Seaborn style
            sns.set_style("whitegrid")
            
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Get the top 10 states by median death count
            state_median_deaths = data_copy.groupby("State")["Deaths"].median().sort_values(ascending=False)
            top_10_states = state_median_deaths.head(10).index.tolist()
            
            # Filter for just those states
            plot_data = data_copy[data_copy["State"].isin(top_10_states)]
            
            # Create the boxplot
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.boxplot(x="State", y="Deaths", data=plot_data, palette="viridis", ax=ax)
            ax.set_title("Distribution of Death Counts Across Top 10 States (by Median)", fontsize=14)
            ax.set_xlabel("State", fontsize=12)
            ax.set_ylabel("Deaths", fontsize=12)
            ax.set_yscale("log")  # Using log scale to better visualize the distribution
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
**Insights from Seaborn Boxplots:**

1. **Statistical Summaries in One Plot**:
   - The central line shows the median
   - The box shows the interquartile range (25th to 75th percentile)
   - The whiskers show the range of non-outlier data
   - Points beyond the whiskers are outliers
   
2. **Comparing Distributions**:
   - Boxplots allow comparison of multiple distributions side by side
   - The log scale makes it possible to compare distributions with different ranges
   - Each state has a different distribution pattern of death counts
   
3. **Advantages of Seaborn**:
   - Professional appearance with minimal code
   - Built-in statistical visualization capabilities
   - Excellent for exploratory data analysis
            """)
            
        with st.expander("4. Heatmap with Annotations"):
            st.markdown("""
### Heatmap with Annotations

**Objective:**  
Create a heatmap to visualize a complex matrix of values.

**Key Points:**
- Heatmaps use color intensity to represent numerical values
- They're excellent for visualizing matrices, correlation tables, and pivot tables
- Annotations add specific values to make the visualization more informative

**Example:**  
Create a heatmap showing death counts across different states and causes:
            """)
            st.code(
'''# Create a heatmap with Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Check if we have cause of death data
if "Cause of death Code" in data_copy.columns:
    # Create a simplified ICD chapter column
    def get_icd_category(code):
        if pd.isna(code):
            return "Unknown"
        first_char = str(code)[0]
        if first_char in "AB":
            return "Infectious"
        elif first_char == "C":
            return "Neoplasms"
        elif first_char == "E":
            return "Endocrine"
        elif first_char == "I":
            return "Circulatory"
        elif first_char == "J":
            return "Respiratory"
        else:
            return "Other"
    
    data_copy["ICD Category"] = data_copy["Cause of death Code"].apply(get_icd_category)
    
    # Create a pivot table for the heatmap
    # Get top 8 states by total deaths for better readability
    top_states = data_copy.groupby("State")["Deaths"].sum().nlargest(8).index.tolist()
    filtered_data = data_copy[data_copy["State"].isin(top_states)]
    
    # Create pivot table: states vs ICD categories
    pivot = pd.pivot_table(
        filtered_data, 
        values="Deaths",
        index="State", 
        columns="ICD Category", 
        aggfunc="sum",
        fill_value=0
    )
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=.5)
    plt.title("Total Deaths by State and Cause Category")
    plt.tight_layout()
    plt.show()
    
else:
    # Alternative: Create a correlation heatmap of numeric columns
    numeric_data = data_copy.select_dtypes(include=['number']).corr()
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, vmin=-1, vmax=1)
    plt.title("Correlation Between Numeric Variables")
    plt.tight_layout()
    plt.show()''', language="python")
            
            # Execute the code with the real data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            if "Cause of death Code" in data_copy.columns:
                # Create a simplified ICD chapter column
                def get_icd_category(code):
                    if pd.isna(code):
                        return "Unknown"
                    first_char = str(code)[0]
                    if first_char in "AB":
                        return "Infectious"
                    elif first_char == "C":
                        return "Neoplasms"
                    elif first_char == "E":
                        return "Endocrine"
                    elif first_char == "I":
                        return "Circulatory"
                    elif first_char == "J":
                        return "Respiratory"
                    else:
                        return "Other"
                
                data_copy["ICD Category"] = data_copy["Cause of death Code"].apply(get_icd_category)
                
                # Create a pivot table for the heatmap
                # Get top 8 states by total deaths for better readability
                top_states = data_copy.groupby("State")["Deaths"].sum().nlargest(8).index.tolist()
                filtered_data = data_copy[data_copy["State"].isin(top_states)]
                
                # Create pivot table: states vs ICD categories
                pivot = pd.pivot_table(
                    filtered_data, 
                    values="Deaths",
                    index="State", 
                    columns="ICD Category", 
                    aggfunc="sum",
                    fill_value=0
                )
                
                # Create the heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=.5, ax=ax)
                ax.set_title("Total Deaths by State and Cause Category")
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                # Try to convert other columns to numeric for correlation
                for col in data_copy.columns:
                    if col not in ["Deaths", "Population"]:
                        try:
                            data_copy[col] = pd.to_numeric(data_copy[col], errors="coerce")
                        except:
                            pass
                        
                # Create correlation of available numeric columns
                numeric_data = data_copy.select_dtypes(include=['number']).corr()
                
                # Create the heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric_data, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, vmin=-1, vmax=1, ax=ax)
                ax.set_title("Correlation Between Numeric Variables")
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("""
**Insights from Heatmaps:**

1. **Color-Coded Intensity**:
   - The color intensity shows the magnitude of the values
   - Annotations provide the exact numbers for precision
   - Patterns and outliers become immediately visible
   
2. **Comparing Multiple Categories**:
   - The heatmap shows relationships between two categorical variables
   - It's easy to spot which combinations have the highest values
   - Patterns across rows or columns reveal trends by state or cause
   
3. **When to Use Heatmaps**:
   - For correlation matrices to see relationships between variables
   - For pivot tables to show summaries across two dimensions
   - When you have a dense dataset that would be hard to visualize with other chart types
            """)
# 🔴 Advanced: Interactive and Custom Visualizations
    with tab3:
        st.markdown("### 🔴 Advanced: Interactive and Custom Visualizations")
        with st.expander("1. Interactive Choropleth Map with Raw Numbers"):
            st.markdown("""
    ### Interactive Choropleth Map with Raw Numbers

    **Objective:**  
    Create an interactive geographic visualization of death counts across states.

    **Key Points:**
    - Choropleth maps use color intensity to show data values by geographic region
    - Interactive features allow users to hover over regions to see exact values
    - They're excellent for showing spatial patterns in your data

    **Example:**  
    Create an interactive choropleth map of total deaths by state:
            """)
            st.code(
    '''# Create a choropleth map with raw death counts
    import plotly.express as px
    import pandas as pd

    # Prepare the data
    data_copy = data.copy()
    data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

    # Calculate total deaths by state
    state_totals = data_copy.groupby("State")["Deaths"].sum().reset_index()

    # Create the choropleth map
    fig = px.choropleth(
        state_totals,
        locations="State",
        locationmode="USA-states",
        color="Deaths",
        scope="usa",
        color_continuous_scale="Viridis",
        hover_name="State",
        hover_data={"Deaths": True},
        labels={"Deaths": "Total Deaths"},
        title="Total Deaths by State (Raw Numbers)"
    )

    fig.update_layout(
        geo=dict(lakecolor="LightBlue"),
        coloraxis_colorbar=dict(title="Deaths")
    )

    fig.show()''', language="python")
            
            # Execute the code with the real data
            import pandas as pd
            import plotly.express as px




            # Step 1: Filter the dataset for Atherosclerotic heart disease deaths
            data1 = data[data['Cause of death'] == 'Atherosclerotic heart disease']

            # Step 2: Drop rows with NaN in the 'Deaths' column and group by state
            data1 = data1.dropna(subset=['Deaths'])
            state_heart_deaths = data1.groupby('State')['Deaths'].sum().reset_index()

            # Step 3: Map state names to abbreviations
            state_abbrev = {
                'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
                'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 
                'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 
                'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 
                'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 
                'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 
                'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 
                'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 
                'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 
                'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 
                'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
            }

            state_heart_deaths['State'] = state_heart_deaths['State'].map(state_abbrev)

            # Step 4: Create the choropleth map
            fig = px.choropleth(
                state_heart_deaths,
                locations='State',  # Column with state abbreviations
                locationmode='USA-states',  # Specifies state-level map
                color='Deaths',  # Column to color by
                scope='usa',
                color_continuous_scale='Reds',  # Optional: set color scale
                title='Atherosclerotic Heart Disease Deaths by State (2022)'
            )

            # Display the map
            st.plotly_chart(fig)
            st.markdown("""
    **Insights from Choropleth Maps:**

    - Geographic patterns are immediately visible
    - States with higher populations generally have higher total death counts
    - The interactive hover tooltip shows the exact value for each state
    - Raw counts don't account for population differences, which is addressed in the next visualization
            """)
            
        with st.expander("2. Choropleth Map with Crude Death Rate"):
            st.markdown("""
    ### Choropleth Map with Crude Death Rate

    **Objective:**  
    Create a map that shows death rates standardized by population size.

    **Key Points:**
    - Crude rates (deaths per 100,000 population) account for population differences
    - This provides a more fair comparison between states of different sizes
    - Using rates instead of raw counts reveals different patterns

    **Example:**  
    Create an interactive choropleth map of crude death rates by state:
            """)
            st.code(
    '''# Create a choropleth map with crude death rates
    import plotly.express as px
    import pandas as pd

    # Prepare the data
    data_copy = data.copy()
    data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
    data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

    # Calculate crude rate by state
    state_data = data_copy.groupby("State").agg({
        "Deaths": "sum",
        "Population": "mean"  # Using mean as population should be same for all records of a state
    }).reset_index()

    # Calculate crude death rate per 100,000 population
    state_data["Crude Rate"] = (state_data["Deaths"] / state_data["Population"]) * 100000

    # Create the choropleth map
    fig = px.choropleth(
        state_data,
        locations="State",
        locationmode="USA-states",
        color="Crude Rate",
        scope="usa",
        color_continuous_scale="Reds",
        hover_name="State",
        hover_data={
            "Deaths": True,
            "Population": True,
            "Crude Rate": ":.2f"
        },
        labels={"Crude Rate": "Deaths per 100,000"},
        title="Crude Death Rate by State"
    )

    fig.update_layout(
        geo=dict(lakecolor="LightBlue"),
        coloraxis_colorbar=dict(title="Deaths per 100,000")
    )

    fig.show()''', language="python")
            
            
            # Step 1: Filter the dataset for Atherosclerotic heart disease deaths
            data1 = data[data['Cause of death'] == 'Atherosclerotic heart disease']

            # Step 2: Drop rows with NaN in the 'Deaths' column and group by state
            data1 = data1.dropna(subset=['Deaths'])
            state_heart_deaths = data1

            # Step 3: Map state names to abbreviations
            state_abbrev = {
                'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
                'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 
                'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 
                'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 
                'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 
                'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 
                'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 
                'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 
                'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 
                'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 
                'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
            }

            state_heart_deaths['State'] = state_heart_deaths['State'].map(state_abbrev)

            state_heart_deaths = data1.groupby("State").agg({
                "Deaths": "sum",
                "Population": "mean"  # Using mean as population should be same for all records of a state
            }).reset_index()
            
            # Calculate crude death rate per 100,000 population
            state_heart_deaths["Crude Rate"] = (state_heart_deaths["Deaths"] / state_heart_deaths["Population"]) * 100000
            
            # Create the choropleth map
            fig = px.choropleth(
                state_heart_deaths,
                locations="State",
                locationmode="USA-states",
                color="Crude Rate",
                scope="usa",
                color_continuous_scale="Reds",
                hover_name="State",
                hover_data={
                    "Deaths": True,
                    "Population": True,
                    "Crude Rate": ":.2f"
                },
                labels={"Crude Rate": "Deaths per 100,000"},
                title="Crude Death Rate by State"
            )
            
            fig.update_layout(
                geo=dict(lakecolor="LightBlue"),
                coloraxis_colorbar=dict(title="Deaths per 100,000")
            )
            
            st.plotly_chart(fig)
            
            
            
            
            
    
            
            st.markdown("""
    **Comparing Raw Counts vs. Crude Rates:**

    1. **Different Stories**:
    - The raw count map often highlights states with larger populations
    - The crude rate map shows the death rate adjusted for population size
    - States with high counts might have low rates, and vice versa
    
    2. **Public Health Implications**:
    - Crude rates help identify states where residents face higher mortality risks
    - They provide a fairer basis for comparing disease burden across regions
    - Population-adjusted rates are crucial for health policy decisions
    
    3. **Data Communication**:
    - Always clearly indicate whether you're showing counts or rates
    - Different audiences may need different metrics (e.g., policymakers vs. researchers)
    - Consider showing both to provide a more complete picture
            """)
            
        with st.expander("3. Custom Multi-Plot Visualization"):
            st.markdown("""
    ### Custom Multi-Plot Visualization

    **Objective:**  
    Create a sophisticated multi-panel visualization that combines different chart types.

    **Key Points:**
    - Complex visualizations can combine multiple chart types for deeper insights
    - Custom styling creates more professional and visually appealing charts
    - These advanced visualizations require more code but provide richer information

    **Example:**  
    Create a dashboard-style visualization with multiple plots:
            """)
            st.code(
    '''# Create a custom multi-plot dashboard
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # Prepare the data
    data_copy = data.copy()
    data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
    data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

    # Calculate state-level summaries
    state_summary = data_copy.groupby("State").agg({
        "Deaths": "sum",
        "Population": "mean"
    }).reset_index()
    state_summary["Death Rate"] = (state_summary["Deaths"] / state_summary["Population"]) * 100000
    state_summary = state_summary.sort_values("Deaths", ascending=False)

    # Get top 10 states by deaths
    top10_states = state_summary.head(10)

    # Set up a complex grid layout
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.5])

    # 1. Bar chart of top 10 states by deaths
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.bar(top10_states["State"], top10_states["Deaths"], color="skyblue")
    ax1.set_title("Top 10 States by Total Deaths", fontsize=14)
    ax1.set_ylabel("Total Deaths", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # 2. Scatter plot of population vs deaths for all states
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(state_summary["Population"], state_summary["Deaths"], 
            s=state_summary["Death Rate"]/10, alpha=0.7, 
            c=state_summary["Death Rate"], cmap="viridis")
    ax2.set_title("Population vs Deaths", fontsize=14)
    ax2.set_xlabel("Population", fontsize=12)
    ax2.set_ylabel("Deaths", fontsize=12)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # 3. Large map/heatmap
    ax3 = fig.add_subplot(gs[1, :])

    # Check if we have cause of death data for the heatmap
    if "Cause of death Code" in data_copy.columns:
        # Create a simplified ICD chapter column
        def get_icd_category(code):
            if pd.isna(code):
                return "Unknown"
            first_char = str(code)[0]
            categories = {
                "A": "Infectious Diseases (A)",
                "B": "Infectious Diseases (B)",
                "C": "Neoplasms",
                "D": "Blood Disorders",
                "E": "Endocrine Disorders",
                "F": "Mental Disorders",
                "G": "Nervous System",
                "I": "Circulatory System",
                "J": "Respiratory System",
                "K": "Digestive System",
                "N": "Genitourinary",
                "R": "Symptoms & Signs",
                "V": "External Causes (V)",
                "W": "External Causes (W)",
                "X": "External Causes (X)",
                "Y": "External Causes (Y)"
            }
            return categories.get(first_char, "Other")
        
        data_copy["ICD Category"] = data_copy["Cause of death Code"].apply(get_icd_category)
        
        # Create a pivot table for the heatmap
        top_states = state_summary.head(15)["State"].tolist()
        filtered_data = data_copy[data_copy["State"].isin(top_states)]
        
        # Create pivot table: states vs ICD categories
        pivot = pd.pivot_table(
            filtered_data, 
            values="Deaths",
            index="State", 
            columns="ICD Category", 
            aggfunc="sum",
            fill_value=0
        )
        
        # Sort the pivot table by total deaths
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=False)
        pivot = pivot.drop("Total", axis=1)
        
        # Create a heatmap
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax3)
        ax3.set_title("Deaths by State and Cause Category", fontsize=16)
        
    else:
        # Alternative: Death rate by state with custom styling
        states = top10_states["State"]
        death_rates = top10_states["Death Rate"]
        
        # Horizontal bar chart of death rates
        colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
        ax3.barh(states, death_rates, color=colors)
        ax3.set_title("Death Rates by State (per 100,000 population)", fontsize=16)
        ax3.set_xlabel("Deaths per 100,000 Population", fontsize=14)
        ax3.grid(axis="x", linestyle="--", alpha=0.7)
        
        # Add value labels to the bars
        for i, v in enumerate(death_rates):
            ax3.text(v + 50, i, f"{v:.1f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()''', language="python")
            
            # Execute the code with the real data
            import matplotlib.gridspec as gridspec
            
            # Prepare the data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
            
            # Calculate state-level summaries
            state_summary = data_copy.groupby("State").agg({
                "Deaths": "sum",
                "Population": "mean"
            }).reset_index()
            state_summary["Death Rate"] = (state_summary["Deaths"] / state_summary["Population"]) * 100000
            state_summary = state_summary.sort_values("Deaths", ascending=False)
            
            # Get top 10 states by deaths
            top10_states = state_summary.head(10)
            
            # Set up a complex grid layout
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.5])
            
            # 1. Bar chart of top 10 states by deaths
            ax1 = fig.add_subplot(gs[0, 0:2])
            ax1.bar(top10_states["State"], top10_states["Deaths"], color="skyblue")
            ax1.set_title("Top 10 States by Total Deaths", fontsize=14)
            ax1.set_ylabel("Total Deaths", fontsize=12)
            ax1.tick_params(axis="x", rotation=45)
            ax1.grid(axis="y", linestyle="--", alpha=0.7)
            
            # 2. Scatter plot of population vs deaths for all states
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.scatter(state_summary["Population"], state_summary["Deaths"], 
                    s=state_summary["Death Rate"]/10, alpha=0.7, 
                    c=state_summary["Death Rate"], cmap="viridis")
            ax2.set_title("Population vs Deaths", fontsize=14)
            ax2.set_xlabel("Population", fontsize=12)
            ax2.set_ylabel("Deaths", fontsize=12)
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.grid(True, linestyle="--", alpha=0.7)
            
            # 3. Large map/heatmap
            ax3 = fig.add_subplot(gs[1, :])
            
            # Check if we have cause of death data for the heatmap
            if "Cause of death Code" in data_copy.columns:
                # Create a simplified ICD chapter column
                def get_icd_category(code):
                    if pd.isna(code):
                        return "Unknown"
                    first_char = str(code)[0]
                    categories = {
                        "A": "Infectious Diseases (A)",
                        "B": "Infectious Diseases (B)",
                        "C": "Neoplasms",
                        "D": "Blood Disorders",
                        "E": "Endocrine Disorders",
                        "F": "Mental Disorders",
                        "G": "Nervous System",
                        "I": "Circulatory System",
                        "J": "Respiratory System",
                        "K": "Digestive System",
                        "N": "Genitourinary",
                        "R": "Symptoms & Signs",
                        "V": "External Causes (V)",
                        "W": "External Causes (W)",
                        "X": "External Causes (X)",
                        "Y": "External Causes (Y)"
                    }
                    return categories.get(first_char, "Other")
                
                data_copy["ICD Category"] = data_copy["Cause of death Code"].apply(get_icd_category)
                
                # Create a pivot table for the heatmap
                top_states = state_summary.head(15)["State"].tolist()
                filtered_data = data_copy[data_copy["State"].isin(top_states)]
                
                # Create pivot table: states vs ICD categories
                pivot = pd.pivot_table(
                    filtered_data, 
                    values="Deaths",
                    index="State", 
                    columns="ICD Category", 
                    aggfunc="sum",
                    fill_value=0
                )
                
                # Sort the pivot table by total deaths
                pivot["Total"] = pivot.sum(axis=1)
                pivot = pivot.sort_values("Total", ascending=False)
                pivot = pivot.drop("Total", axis=1)
                
                # Create a heatmap
                sns.heatmap(pivot, cmap="YlOrRd", ax=ax3)
                ax3.set_title("Deaths by State and Cause Category", fontsize=16)
                
            else:
                # Alternative: Death rate by state with custom styling
                states = top10_states["State"]
                death_rates = top10_states["Death Rate"]
                
                # Horizontal bar chart of death rates
                colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
                ax3.barh(states, death_rates, color=colors)
                ax3.set_title("Death Rates by State (per 100,000 population)", fontsize=16)
                ax3.set_xlabel("Deaths per 100,000 Population", fontsize=14)
                ax3.grid(axis="x", linestyle="--", alpha=0.7)
                
                # Add value labels to the bars
                for i, v in enumerate(death_rates):
                    ax3.text(v + 50, i, f"{v:.1f}", va="center", fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
    **Benefits of Multi-Panel Visualizations:**

    1. **Comprehensive Data Storytelling**:
    - Combine different perspectives on your data in a single view
    - Show relationships between different metrics
    - Create a more complete narrative about your data
    
    2. **Space Efficiency**:
    - Present more information in a single figure
    - Allow direct comparisons between different visualizations
    - Create dashboard-like views for reporting
    
    3. **Professional Presentation**:
    - Custom formatting creates polished visualizations for reports and publications
    - GridSpec layout system provides precise control over figure composition
    - Custom color schemes and annotations enhance readability and aesthetics
    
    4. **When to Use**:
    - For final presentations and reports
    - When you need to show multiple related analyses together
    - To create executive dashboards summarizing key findings
            """)
            
        with st.expander("4. Interactive Visualization with User Controls"):
            st.markdown("""
    ### Interactive Visualization with User Controls

    **Objective:**  
    Create an interactive visualization that allows users to explore different aspects of the data.

    **Key Points:**
    - Interactive visualizations enable users to explore data themselves
    - Dynamic filtering and faceting reveal patterns that might be missed in static visualizations
    - These advanced visualizations help users engage with your data and draw their own conclusions

    **Example:**  
    Build an interactive visualization with Plotly Express:
            """)
            st.code(
    '''# Create an interactive visualization with dynamic controls
    import plotly.express as px
    import pandas as pd

    # Prepare the data
    data_copy = data.copy()
    data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
    data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

    # Calculate state-level statistics
    state_stats = data_copy.groupby("State").agg({
        "Deaths": ["sum", "mean", "median", "std"],
        "Population": "mean"
    }).reset_index()

    # Flatten the MultiIndex columns
    state_stats.columns = ["State", "Total_Deaths", "Mean_Deaths", "Median_Deaths", "Std_Deaths", "Population"]

    # Add death rate
    state_stats["Death_Rate"] = (state_stats["Total_Deaths"] / state_stats["Population"]) * 100000

    # Create an interactive scatter plot with multiple dimensions
    fig = px.scatter(
        state_stats,
        x="Population",
        y="Total_Deaths",
        size="Death_Rate",
        color="Death_Rate",
        hover_name="State",
        hover_data={
            "Population": True,
            "Total_Deaths": True,
            "Mean_Deaths": ":.1f",
            "Median_Deaths": ":.1f",
            "Death_Rate": ":.2f",
            "Std_Deaths": ":.2f"
        },
        log_x=True,
        log_y=True,
        size_max=50,
        color_continuous_scale="Viridis",
        title="State Death Statistics (Hover for Details)"
    )

    # Add custom hover template
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
        "Population: %{x:,.0f}<br>" +
        "Total Deaths: %{y:,.0f}<br>" +
        "Death Rate: %{marker.color:.1f} per 100,000<br>" +
        "Mean Deaths: %{customdata[2]:.1f}<br>" +
        "Median Deaths: %{customdata[3]:.1f}<br>" +
        "Std Dev: %{customdata[5]:.1f}<extra></extra>"
    )

    # Add annotations and styling
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis_title="Population (Log Scale)",
        yaxis_title="Total Deaths (Log Scale)",
        coloraxis_colorbar=dict(title="Death Rate<br>per 100,000"),
        annotations=[
            dict(
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                text="Bubble size represents death rate per 100,000 population",
                showarrow=False,
                font=dict(size=12)
            )
        ],
        height=700
    )

    fig.show()''', language="python")
            
            # Execute the code with the real data
            # Prepare the data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
            
            # Calculate state-level statistics
            state_stats = data_copy.groupby("State").agg({
                "Deaths": ["sum", "mean", "median", "std"],
                "Population": "mean"
            }).reset_index()
            
            # Flatten the MultiIndex columns
            state_stats.columns = ["State", "Total_Deaths", "Mean_Deaths", "Median_Deaths", "Std_Deaths", "Population"]
            
            # Add death rate
            state_stats["Death_Rate"] = (state_stats["Total_Deaths"] / state_stats["Population"]) * 100000
            
            # Create an interactive scatter plot with multiple dimensions
            fig = px.scatter(
                state_stats,
                x="Population",
                y="Total_Deaths",
                size="Death_Rate",
                color="Death_Rate",
                hover_name="State",
                hover_data={
                    "Population": True,
                    "Total_Deaths": True,
                    "Mean_Deaths": ":.1f",
                    "Median_Deaths": ":.1f",
                    "Death_Rate": ":.2f",
                    "Std_Deaths": ":.2f"
                },
                log_x=True,
                log_y=True,
                size_max=50,
                color_continuous_scale="Viridis",
                title="State Death Statistics (Hover for Details)"
            )
            
            # Add custom hover template
            fig.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>" +
                "Population: %{x:,.0f}<br>" +
                "Total Deaths: %{y:,.0f}<br>" +
                "Death Rate: %{marker.color:.1f} per 100,000<br>" +
                "Mean Deaths: %{customdata[2]:.1f}<br>" +
                "Median Deaths: %{customdata[3]:.1f}<br>" +
                "Std Dev: %{customdata[5]:.1f}<extra></extra>"
            )
            
            # Add annotations and styling
            fig.update_layout(
                hoverlabel=dict(bgcolor="white", font_size=12),
                xaxis_title="Population (Log Scale)",
                yaxis_title="Total Deaths (Log Scale)",
                coloraxis_colorbar=dict(title="Death Rate<br>per 100,000"),
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.15,
                        xref="paper",
                        yref="paper",
                        text="Bubble size represents death rate per 100,000 population",
                        showarrow=False,
                        font=dict(size=12)
                    )
                ],
                height=700
            )
            
            st.plotly_chart(fig)
            
            st.markdown("""
    **Benefits of Interactive Visualization:**

    1. **Enhanced Data Exploration**:
    - Users can hover over points to see detailed information
    - Multi-dimensional visualization shows relationships between several variables
    - Log scales make it possible to compare states of vastly different sizes
    
    2. **Information Density**:
    - Multiple dimensions of data in a single visualization (position, size, color)
    - Hover tooltips provide detailed statistics without cluttering the display
    - Interactive elements encourage deeper exploration of the data

    3. **User Engagement**:
    - Interactive visualizations invite users to explore the data themselves
    - Discovery-driven approach leads to deeper insights and better understanding
    - Engaging visualizations make complex data more accessible to non-technical audiences
            """)            
            


























elif st.session_state.page == "step7_statistical_analysis":
    st.header("📊 Step 7 – Statistical Analysis")
    # Back navigation button
    if st.button("← Back to Knowledge Graph", key="back_btn_step7"):
        st.session_state.page = "graph"
        st.session_state.reset_guidance = True
        st.rerun()
        
    st.subheader("Choose Your Level")
    tab1, tab2, tab3 = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"])
    
    # -------------------------------------------------------------------------
    # BEGINNER LEVEL: Descriptive Statistics
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### 🟢 Beginner Level: Descriptive Statistics")
        
        with st.expander("📊 Basic Summary Statistics"):
            st.markdown("""
### Summary Statistics with pandas

**Objective:**  
Learn how to calculate and interpret basic descriptive statistics for your data.

**Key Concepts:**
- Mean: The average value of a variable
- Median: The middle value when data is sorted
- Standard Deviation: A measure of data spread
- Min/Max: The smallest and largest values
- Percentiles: Values below which a certain percentage of observations fall
            """)
            
            st.code('''# Get basic statistics for numerical columns
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

# Display the summary statistics
stats = data_copy.describe()
print(stats)

# Look at specific statistics
print(f"Mean deaths: {data_copy['Deaths'].mean():.2f}")
print(f"Median deaths: {data_copy['Deaths'].median():.2f}")
print(f"Standard deviation: {data_copy['Deaths'].std():.2f}")
print(f"Min deaths: {data_copy['Deaths'].min():.2f}")
print(f"Max deaths: {data_copy['Deaths'].max():.2f}")''', language="python")
            
            # Execute the code with the real data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
            
            stats = data_copy.describe()
            st.dataframe(stats)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Deaths", f"{data_copy['Deaths'].mean():.2f}")
                st.metric("Median Deaths", f"{data_copy['Deaths'].median():.2f}")
                st.metric("Standard Deviation", f"{data_copy['Deaths'].std():.2f}")
            with col2:
                st.metric("Minimum Deaths", f"{data_copy['Deaths'].min():.2f}")
                st.metric("Maximum Deaths", f"{data_copy['Deaths'].max():.2f}")
                st.metric("Death Count Range", f"{data_copy['Deaths'].max() - data_copy['Deaths'].min():.2f}")
            
            st.markdown("""
**Insights:**
- The mean and median give you two different measures of central tendency
- A large difference between mean and median suggests a skewed distribution
- The standard deviation tells you how spread out your data is
- Min/max values help identify the range of your data
            """)
            
        with st.expander("📈 Distribution Analysis"):
            st.markdown("""
### Understanding Data Distributions

**Objective:**  
Analyze the shape and characteristics of your data's distribution.

**Key Concepts:**
- Skewness: Measure of the asymmetry of a distribution
- Kurtosis: Measure of the "tailedness" of a distribution
- Histograms: Visual representation of data distribution
- Density plots: Smoothed version of a histogram
            """)
            
            st.code('''# Calculate skewness and kurtosis
import scipy.stats as stats

data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Calculate skewness and kurtosis
skewness = stats.skew(data_copy["Deaths"].dropna())
kurtosis = stats.kurtosis(data_copy["Deaths"].dropna())

print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

# Interpret skewness
if skewness > 0.5:
    print("The distribution is positively skewed (right tail)")
elif skewness < -0.5:
    print("The distribution is negatively skewed (left tail)")
else:
    print("The distribution is approximately symmetric")

# Interpret kurtosis
if kurtosis > 0.5:
    print("The distribution is leptokurtic (heavy-tailed)")
elif kurtosis < -0.5:
    print("The distribution is platykurtic (light-tailed)")
else:
    print("The distribution is approximately mesokurtic (normal-tailed)")

# Create a histogram with density curve
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data_copy["Deaths"].dropna(), kde=True, bins=30)
plt.title("Distribution of Deaths")
plt.xlabel("Deaths")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()''', language="python")
            
            # Execute the code with the real data
            import scipy.stats as stats
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Calculate skewness and kurtosis
            skewness = stats.skew(data_copy["Deaths"].dropna())
            kurtosis = stats.kurtosis(data_copy["Deaths"].dropna())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Skewness", f"{skewness:.4f}")
                # Interpret skewness
                if skewness > 0.5:
                    st.info("The distribution is positively skewed (right tail)")
                elif skewness < -0.5:
                    st.info("The distribution is negatively skewed (left tail)")
                else:
                    st.info("The distribution is approximately symmetric")
            
            with col2:
                st.metric("Kurtosis", f"{kurtosis:.4f}")
                # Interpret kurtosis
                if kurtosis > 0.5:
                    st.info("The distribution is leptokurtic (heavy-tailed)")
                elif kurtosis < -0.5:
                    st.info("The distribution is platykurtic (light-tailed)")
                else:
                    st.info("The distribution is approximately mesokurtic (normal-tailed)")
            
            # Create a histogram with density curve
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data_copy["Deaths"].dropna(), kde=True, bins=30, ax=ax)
            ax.set_title("Distribution of Deaths")
            ax.set_xlabel("Deaths")
            ax.set_ylabel("Frequency")
            ax.grid(True, linestyle="--", alpha=0.7)
            st.pyplot(fig)
            
            st.markdown("""
**Interpreting Distribution Statistics:**

1. **Skewness:**
   - Positive skewness (> 0): Distribution has a long right tail
   - Negative skewness (< 0): Distribution has a long left tail
   - Zero skewness: Distribution is symmetric

2. **Kurtosis:**
   - Positive kurtosis: Distribution has heavier tails than a normal distribution
   - Negative kurtosis: Distribution has lighter tails than a normal distribution
   - Zero kurtosis: Distribution has tails similar to a normal distribution

3. **What This Tells Us:**
   - Many real-world datasets (like deaths) are positively skewed
   - Positive skewness often suggests a "natural limit" on the low end
   - Heavy tails indicate more extreme values than would be expected in a normal distribution
            """)
            
        with st.expander("🔍 Group-Level Statistics"):
            st.markdown("""
### Comparing Statistics Across Groups

**Objective:**  
Calculate and compare summary statistics across different categories or groups.

**Key Concepts:**
- Group by: Split data into groups for aggregation
- Aggregation: Calculate statistics for each group
- Comparison: Identify differences across groups
            """)
            
            st.code('''# Calculate statistics by state
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Group by State and calculate multiple statistics
state_stats = data_copy.groupby("State")["Deaths"].agg([
    ("Mean", "mean"),
    ("Median", "median"),
    ("Std Dev", "std"),
    ("Min", "min"),
    ("Max", "max"),
    ("Count", "count")
]).round(2)

# Sort by mean deaths in descending order
state_stats = state_stats.sort_values("Mean", ascending=False)

# Display the top 10 states
print(state_stats.head(10))

# Create a bar chart comparing means
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
state_stats["Mean"].head(10).plot(kind="bar")
plt.title("Average Deaths by State (Top 10)")
plt.xlabel("State")
plt.ylabel("Mean Deaths")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Group by State and calculate multiple statistics
            state_stats = data_copy.groupby("State")["Deaths"].agg([
                ("Mean", "mean"),
                ("Median", "median"),
                ("Std Dev", "std"),
                ("Min", "min"),
                ("Max", "max"),
                ("Count", "count")
            ]).round(2)
            
            # Sort by mean deaths in descending order
            state_stats = state_stats.sort_values("Mean", ascending=False)
            
            # Display the top 10 states
            st.dataframe(state_stats.head(10))
            
            # Create a bar chart comparing means
            fig, ax = plt.subplots(figsize=(12, 6))
            state_stats["Mean"].head(10).plot(kind="bar", ax=ax)
            ax.set_title("Average Deaths by State (Top 10)")
            ax.set_xlabel("State")
            ax.set_ylabel("Mean Deaths")
            plt.xticks(rotation=45)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
**Insights from Group Comparisons:**

1. **Identify Patterns:**
   - Some states have higher average death counts than others
   - The variation (standard deviation) differs by state
   - States with higher means might have different distributions

2. **Practical Applications:**
   - Compare outcomes across different geographic regions
   - Identify areas for targeted interventions
   - Find potential outliers or anomalies within specific groups

3. **Considerations:**
   - Higher means could be related to population size differences
   - Sample counts per group can affect reliability of statistics
   - Further investigation would be needed to determine causes
            """)
    
    # -------------------------------------------------------------------------
    # INTERMEDIATE LEVEL: Hypothesis Testing
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 🟡 Intermediate Level: Hypothesis Testing")
        
        with st.expander("🔬 Understanding Hypothesis Tests"):
            st.markdown("""
### Introduction to Hypothesis Testing

**Objective:**  
Learn the fundamentals of hypothesis testing and statistical significance.

**Key Concepts:**
- Null Hypothesis (H₀): Assumes no effect or difference
- Alternative Hypothesis (Hₐ): Assumes an effect or difference exists
- p-value: Probability of observing results at least as extreme as current data if null hypothesis is true
- Statistical Significance: Typically a p-value < 0.05 is considered significant
            """)
            
            st.code('''# Simple hypothesis testing example
import scipy.stats as stats
import numpy as np
import pandas as pd

# Example: Test if the mean deaths are statistically different between two states
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Select two states to compare (using first two states in the dataset for example)
available_states = data_copy["State"].unique()
if len(available_states) >= 2:
    state1 = available_states[0]
    state2 = available_states[1]
    
    # Get death counts for each state
    state1_deaths = data_copy[data_copy["State"] == state1]["Deaths"].dropna()
    state2_deaths = data_copy[data_copy["State"] == state2]["Deaths"].dropna()
    
    # Perform independent samples t-test
    t_stat, p_value = stats.ttest_ind(state1_deaths, state2_deaths, equal_var=False)
    
    # Display results
    print(f"Comparing {state1} vs {state2}")
    print(f"Mean deaths in {state1}: {state1_deaths.mean():.2f}")
    print(f"Mean deaths in {state2}: {state2_deaths.mean():.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Interpret the result
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: The difference in mean deaths between {state1} and {state2} is statistically significant (p < {alpha}).")
        print("We reject the null hypothesis that the means are equal.")
    else:
        print(f"Result: The difference in mean deaths between {state1} and {state2} is not statistically significant (p >= {alpha}).")
        print("We fail to reject the null hypothesis that the means are equal.")
else:
    print("Not enough states in the dataset to perform comparison.")''', language="python")
            
            # Execute the code with the real data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Select two states to compare (using first two states in the dataset for example)
            available_states = data_copy["State"].unique()
            if len(available_states) >= 2:
                state1 = available_states[0]
                state2 = available_states[1]
                
                # Get death counts for each state
                state1_deaths = data_copy[data_copy["State"] == state1]["Deaths"].dropna()
                state2_deaths = data_copy[data_copy["State"] == state2]["Deaths"].dropna()
                
                # Perform independent samples t-test
                t_stat, p_value = stats.ttest_ind(state1_deaths, state2_deaths, equal_var=False)
                
                # Display results in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(f"Mean deaths in {state1}", f"{state1_deaths.mean():.2f}")
                    st.metric("Sample size", f"{len(state1_deaths)}")
                
                with col2:
                    st.metric(f"Mean deaths in {state2}", f"{state2_deaths.mean():.2f}")
                    st.metric("Sample size", f"{len(state2_deaths)}")
                
                st.metric("t-statistic", f"{t_stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                
                # Interpret the result
                alpha = 0.05
                if p_value < alpha:
                    st.success(f"Result: The difference in mean deaths between {state1} and {state2} is statistically significant (p < {alpha}).")
                    st.write("We reject the null hypothesis that the means are equal.")
                else:
                    st.info(f"Result: The difference in mean deaths between {state1} and {state2} is not statistically significant (p >= {alpha}).")
                    st.write("We fail to reject the null hypothesis that the means are equal.")
            else:
                st.warning("Not enough states in the dataset to perform comparison.")
            
            st.markdown("""
**Understanding Hypothesis Testing Results:**

1. **The Null Hypothesis (H₀):**
   - Assumes there is no difference between the mean deaths of the two states
   - This is what we're testing against

2. **The Alternative Hypothesis (Hₐ):**
   - Assumes there is a difference between the mean deaths of the two states

3. **Interpreting p-values:**
   - p < 0.05: Statistically significant difference (reject H₀)
   - p ≥ 0.05: Not enough evidence to claim a difference (fail to reject H₀)

4. **Practical Significance:**
   - Statistical significance doesn't always mean practical importance
   - Consider the actual difference in means alongside the p-value
   - Sample size can affect significance (very large samples can make even tiny differences significant)
            """)
            
        with st.expander("🔄 Correlation Analysis"):
            st.markdown("""
### Correlation Analysis

**Objective:**  
Measure and interpret the strength and direction of relationships between variables.

**Key Concepts:**
- Pearson Correlation: Measures linear relationship between two variables (-1 to +1)
- Spearman Correlation: Measures monotonic relationship, less sensitive to outliers
- Correlation Matrix: Shows all pairwise correlations among variables
- Correlation ≠ Causation: Correlation doesn't imply a causal relationship
            """)
            
            st.code('''# Perform correlation analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for correlation analysis
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

# Calculate summary statistics by state
state_summary = data_copy.groupby("State").agg({
    "Deaths": "sum",
    "Population": "mean"
}).reset_index()

# Calculate death rate
state_summary["Death_Rate"] = (state_summary["Deaths"] / state_summary["Population"]) * 100000

# Calculate correlation coefficients
pearson_corr = state_summary["Population"].corr(state_summary["Deaths"])
spearman_corr = state_summary["Population"].corr(state_summary["Deaths"], method="spearman")

print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
print(f"Spearman correlation coefficient: {spearman_corr:.4f}")

# Interpret the correlation
if abs(pearson_corr) < 0.3:
    strength = "weak"
elif abs(pearson_corr) < 0.7:
    strength = "moderate"
else:
    strength = "strong"

if pearson_corr > 0:
    direction = "positive"
else:
    direction = "negative"

print(f"Interpretation: There is a {strength}, {direction} correlation between population and total deaths.")

# Create a scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x="Population", y="Deaths", data=state_summary)
plt.title(f"Correlation between Population and Deaths (r = {pearson_corr:.4f})")
plt.xlabel("Population")
plt.ylabel("Total Deaths")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()''', language="python")
            
            # Execute the code with the real data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
            
            # Calculate summary statistics by state
            state_summary = data_copy.groupby("State").agg({
                "Deaths": "sum",
                "Population": "mean"
            }).reset_index()
            
            # Calculate death rate
            state_summary["Death_Rate"] = (state_summary["Deaths"] / state_summary["Population"]) * 100000
            
            # Calculate correlation coefficients
            pearson_corr = state_summary["Population"].corr(state_summary["Deaths"])
            spearman_corr = state_summary["Population"].corr(state_summary["Deaths"], method="spearman")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pearson correlation", f"{pearson_corr:.4f}")
            with col2:
                st.metric("Spearman correlation", f"{spearman_corr:.4f}")
            
            # Interpret the correlation
            if abs(pearson_corr) < 0.3:
                strength = "weak"
            elif abs(pearson_corr) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"
            
            if pearson_corr > 0:
                direction = "positive"
            else:
                direction = "negative"
            
            st.info(f"Interpretation: There is a {strength}, {direction} correlation between population and total deaths.")
            
            # Create a scatter plot with regression line
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x="Population", y="Deaths", data=state_summary, ax=ax)
            ax.set_title(f"Correlation between Population and Deaths (r = {pearson_corr:.4f})")
            ax.set_xlabel("Population")
            ax.set_ylabel("Total Deaths")
            ax.grid(True, linestyle="--", alpha=0.7)
            st.pyplot(fig)
            
            # Create correlation matrix if we have more numeric columns
            st.markdown("### Correlation Matrix")
            st.write("Let's look at correlations between all numerical variables:")
            
            # Try to convert any other columns to numeric
            numeric_cols = ["Deaths", "Population"]
            for col in data.columns:
                if col not in numeric_cols:
                    try:
                        data_copy[col] = pd.to_numeric(data_copy[col], errors="coerce")
                        if not data_copy[col].isna().all():  # If not all values are NaN
                            numeric_cols.append(col)
                    except:
                        pass
            
            # Create correlation matrix for numeric columns
            if len(numeric_cols) > 2:
                # Group by state to get one row per state
                state_data = data_copy.groupby("State")[numeric_cols].mean().reset_index()
                
                # Calculate correlation matrix
                corr_matrix = state_data.drop("State", axis=1).corr()
                
                # Display correlation matrix
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
                ax.set_title("Correlation Matrix of Numeric Variables")
                st.pyplot(fig)
            else:
                st.write("Not enough numeric columns for a correlation matrix.")
            
            st.markdown("""
**Interpreting Correlation Results:**

1. **Correlation Strength:**
   - Close to ±1: Strong relationship
   - Around ±0.5: Moderate relationship
   - Close to 0: Weak or no linear relationship

2. **Correlation Direction:**
   - Positive: Variables increase together
   - Negative: As one variable increases, the other decreases

3. **Pearson vs. Spearman:**
   - Pearson measures linear relationships
   - Spearman measures monotonic relationships (direction matters, but not necessarily linearly)
   - Differences between them can indicate non-linear relationships

4. **Important Considerations:**
   - Correlation doesn't imply causation
   - Outliers can significantly affect Pearson correlations
   - Always visualize your data alongside correlation coefficients
   - Correlation only detects linear (Pearson) or monotonic (Spearman) relationships
            """)
            
        with st.expander("📊 ANOVA: Comparing Multiple Groups"):
            st.markdown("""
### Analysis of Variance (ANOVA)

**Objective:**  
Compare means across multiple groups to determine if there are statistically significant differences.

**Key Concepts:**
- ANOVA: Analysis of Variance tests if means differ across three or more groups
- F-statistic: Ratio of between-group variance to within-group variance
- p-value: Determines if differences are statistically significant
- Post-hoc tests: Determine which specific groups differ (if ANOVA is significant)
            """)
            
            st.code('''# Perform one-way ANOVA to compare deaths across multiple states
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Prepare data
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")

# Select top states by number of records for comparison
top_states = data_copy["State"].value_counts().head(5).index.tolist()
filtered_data = data_copy[data_copy["State"].isin(top_states)]

# Create lists of deaths for each state (for scipy's f_oneway)
death_lists = [filtered_data[filtered_data["State"] == state]["Deaths"].dropna() for state in top_states]

# Only proceed if each state has data
if all(len(deaths) > 0 for deaths in death_lists):
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*death_lists)
    
    # Display results
    print(f"ANOVA Results for Death Counts across {len(top_states)} states:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Interpret the result
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: There are statistically significant differences in mean deaths between the states (p < {alpha}).")
        
        # Create a DataFrame for post-hoc tests
        posthoc_data = []
        for i, state in enumerate(top_states):
            for death in death_lists[i]:
                posthoc_data.append({"State": state, "Deaths": death})
        
        posthoc_df = pd.DataFrame(posthoc_data)
        
        # Perform Tukey's HSD test
        tukey_result = pairwise_tukeyhsd(posthoc_df["Deaths"], posthoc_df["State"], alpha=0.05)
        print("\nTukey's HSD Test Results:")
        print(tukey_result)
    else:
        print(f"Result: There are no statistically significant differences in mean deaths between states (p >= {alpha}).")
    
    # Visualize the comparison with a box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="State", y="Deaths", data=filtered_data)
    plt.title("Comparison of Death Counts Across States")
    plt.xlabel("State")
    plt.ylabel("Deaths")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data for ANOVA. Some states have no death records.")''', language="python")
            
            # Execute the code with the real data
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            
            # Prepare data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            
            # Select top states by number of records for comparison
            top_states = data_copy["State"].value_counts().head(5).index.tolist()
            filtered_data = data_copy[data_copy["State"].isin(top_states)]
            
            # Create lists of deaths for each state (for scipy's f_oneway)
            death_lists = [filtered_data[filtered_data["State"] == state]["Deaths"].dropna() for state in top_states]
            
            # Only proceed if each state has data
            if all(len(deaths) > 0 for deaths in death_lists):
                # Perform one-way ANOVA
                f_stat, p_value = stats.f_oneway(*death_lists)
                
                # Display results
                st.write(f"ANOVA Results for Death Counts across {len(top_states)} states:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("F-statistic", f"{f_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                
                # Interpret the result
                alpha = 0.05
                if p_value < alpha:
                    st.success(f"Result: There are statistically significant differences in mean deaths between the states (p < {alpha}).")
                    
                    # Create a DataFrame for post-hoc tests
                    posthoc_data = []
                    for i, state in enumerate(top_states):
                        for death in death_lists[i]:
                            posthoc_data.append({"State": state, "Deaths": death})
                    
                    posthoc_df = pd.DataFrame(posthoc_data)
                    
                    # Perform Tukey's HSD test
                    tukey_result = pairwise_tukeyhsd(posthoc_df["Deaths"], posthoc_df["State"], alpha=0.05)
                    
                    st.write("\nTukey's HSD Test Results:")
                    # Convert Tukey result to DataFrame
                    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                                           columns=tukey_result._results_table.data[0])
                    st.dataframe(tukey_df)
                else:
                    st.info(f"Result: There are no statistically significant differences in mean deaths between states (p >= {alpha}).")
                
                # Visualize the comparison with a box plot
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(x="State", y="Deaths", data=filtered_data, ax=ax)
                ax.set_title("Comparison of Death Counts Across States")
                ax.set_xlabel("State")
                ax.set_ylabel("Deaths")
                ax.grid(True, linestyle="--", alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Insufficient data for ANOVA. Some states have no death records.")
            
            st.markdown("""
**Understanding ANOVA Results:**

1. **When to Use ANOVA:**
   - When comparing means across three or more groups
   - For determining if at least one group's mean differs from others

2. **Interpreting Results:**
   - F-statistic: Higher values suggest greater between-group differences
   - p-value < 0.05: Indicates statistically significant differences between at least some groups
   - p-value ≥ 0.05: No statistical evidence of differences between groups

3. **Post-hoc Tests (Tukey's HSD):**
   - Only performed if ANOVA is significant
   - Identifies which specific group pairs have significant differences
   - Helps control for multiple comparison problems

4. **Practical Considerations:**
   - ANOVA assumes normal distribution within groups
   - Assumes homogeneity of variance across groups
   - For non-normal data, consider non-parametric alternatives (Kruskal-Wallis test)
            """)
             # Add ANOVA for Crude Rates
            st.markdown("### ANOVA for Crude Death Rates")
            st.code('''# Perform ANOVA on crude death rates
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data - calculate crude rates by state
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
data_copy["Crude_Rate"] = (data_copy["Deaths"] / data_copy["Population"]) * 100000

# Select top states by number of records
top_states = data_copy["State"].value_counts().head(5).index.tolist()
filtered_data = data_copy[data_copy["State"].isin(top_states)]

# Create lists of crude rates for each state
crude_rate_lists = [filtered_data[filtered_data["State"] == state]["Crude_Rate"].dropna() for state in top_states]

# Perform ANOVA if we have data
if all(len(rates) > 0 for rates in crude_rate_lists):
    f_stat, p_value = stats.f_oneway(*crude_rate_lists)
    
    print(f"ANOVA Results for Crude Death Rates across {len(top_states)} states:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Visualize with boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="State", y="Crude_Rate", data=filtered_data)
    plt.title("Comparison of Crude Death Rates Across States")
    plt.xlabel("State")
    plt.ylabel("Deaths per 100,000 Population")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()''', language="python")
            
            # Try to execute crude rate ANOVA
            try:
                # Prepare data - calculate crude rates by state
                data_copy = data.copy()
                data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
                data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
                
                # Calculate crude rate directly if not available
                if "Crude Rate" in data_copy.columns:
                    data_copy["Crude_Rate"] = pd.to_numeric(data_copy["Crude Rate"], errors="coerce")
                else:
                    data_copy["Crude_Rate"] = (data_copy["Deaths"] / data_copy["Population"]) * 100000
                
                # Select top states by number of records
                top_states = data_copy["State"].value_counts().head(5).index.tolist()
                filtered_data = data_copy[data_copy["State"].isin(top_states)]
                
                # Create lists of crude rates for each state
                crude_rate_lists = [filtered_data[filtered_data["State"] == state]["Crude_Rate"].dropna() for state in top_states]
                
                # Perform ANOVA if we have data
                if all(len(rates) > 0 for rates in crude_rate_lists):
                    f_stat, p_value = stats.f_oneway(*crude_rate_lists)
                    
                    st.write(f"### ANOVA Results for Crude Death Rates across {len(top_states)} states:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("F-statistic", f"{f_stat:.4f}")
                    with col2:
                        st.metric("p-value", f"{p_value:.4f}")
                    
                    # Interpret result
                    if p_value < 0.05:
                        st.success("There are significant differences in crude death rates between the states.")
                    else:
                        st.info("The crude death rates do not differ significantly between the states.")
                    
                    # Visualize with boxplot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.boxplot(x="State", y="Crude_Rate", data=filtered_data, ax=ax)
                    ax.set_title("Comparison of Crude Death Rates Across States")
                    ax.set_xlabel("State")
                    ax.set_ylabel("Deaths per 100,000 Population")
                    ax.grid(True, linestyle="--", alpha=0.7)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Insufficient data for crude rate ANOVA. Some states have no crude rate data.")
            except Exception as e:
                st.error(f"Error performing crude rate ANOVA: {e}")
    
    # -------------------------------------------------------------------------
    # ADVANCED LEVEL: Regression and Modeling
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### 🔴 Advanced Level: Regression and Modeling")
        
        with st.expander("📈 Simple Linear Regression"):
            st.markdown("""
### Simple Linear Regression

**Objective:**  
Model the relationship between two continuous variables using a straight line.

**Key Concepts:**
- Dependent variable (y): The outcome we're trying to predict
- Independent variable (x): The predictor or feature used for prediction
- Coefficient (slope): The change in y for a one-unit change in x
- Intercept: The value of y when x equals zero
- R-squared: Proportion of variance in y explained by the model
            """)
            
            st.code('''# Perform simple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Prepare the data
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

# Calculate state-level statistics
state_summary = data_copy.groupby("State").agg({
    "Deaths": "sum",
    "Population": "mean"
}).reset_index()

# Remove rows with missing values
state_summary = state_summary.dropna(subset=["Deaths", "Population"])

# Define X (predictor) and y (outcome)
X = state_summary["Population"].values.reshape(-1, 1)
y = state_summary["Deaths"].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Get model coefficients and metrics
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate error metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# Display results
print("Simple Linear Regression: Population vs. Deaths")
print(f"Equation: Deaths = {slope:.6f} × Population + {intercept:.2f}")
print(f"R-squared: {r_squared:.4f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Interpret the model
print("\nInterpretation:")
print(f"For each additional person in the population, there are approximately {slope:.6f} additional deaths.")
print(f"The model explains {r_squared:.1%} of the variance in death counts across states.")

# Create a scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Population", y="Deaths", data=state_summary)
plt.plot(X, y_pred, color='red', linewidth=2)
plt.title("Linear Regression: Population vs. Deaths")
plt.xlabel("Population")
plt.ylabel("Total Deaths")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Prepare the data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
            
            # Calculate state-level statistics
            state_summary = data_copy.groupby("State").agg({
                "Deaths": "sum",
                "Population": "mean"
            }).reset_index()
            
            # Remove rows with missing values
            state_summary = state_summary.dropna(subset=["Deaths", "Population"])
            
            # Define X (predictor) and y (outcome)
            X = state_summary["Population"].values.reshape(-1, 1)
            y = state_summary["Deaths"].values
            
            # Create and fit the model
            model = LinearRegression()
            model.fit(X, y)
            
            # Get model coefficients and metrics
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(X, y)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate error metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Display results
            st.write("Simple Linear Regression: Population vs. Deaths")
            st.info(f"Equation: Deaths = {slope:.6f} × Population + {intercept:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R-squared", f"{r_squared:.4f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            
            st.write("\nInterpretation:")
            st.write(f"For each additional person in the population, there are approximately {slope:.6f} additional deaths.")
            st.write(f"The model explains {r_squared:.1%} of the variance in death counts across states.")
            
            # Create a scatter plot with regression line
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x="Population", y="Deaths", data=state_summary, ax=ax)
            ax.plot(X, y_pred, color='red', linewidth=2)
            ax.set_title("Linear Regression: Population vs. Deaths")
            ax.set_xlabel("Population")
            ax.set_ylabel("Total Deaths")
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
**Understanding Linear Regression Results:**

1. **The Regression Equation:**
   - `y = mx + b` where `y` is Deaths, `x` is Population, `m` is the slope, and `b` is the intercept
   - The slope tells you how many additional deaths per additional person
   - The intercept represents the estimated deaths when population is zero (often not meaningful)

2. **Assessing Model Quality:**
   - R-squared: Percentage of variance explained (0 to 1, higher is better)
   - RMSE (Root Mean Squared Error): Average prediction error in the same units as the dependent variable

3. **Practical Implications:**
   - The model can be used to predict deaths based on population
   - The slope quantifies the relationship between population and deaths
   - A significant positive relationship confirms that larger states tend to have more deaths

4. **Assumptions to Consider:**
   - Linearity: The relationship between variables is linear
   - Independence: Observations are independent
   - Homoscedasticity: Variance of errors is constant
   - Normality: Errors are normally distributed
            """)
            
        with st.expander("📉 Multiple Linear Regression"):
            st.markdown("""
### Multiple Linear Regression

**Objective:**  
Model the relationship between an outcome variable and multiple predictor variables.

**Key Concepts:**
- Multiple predictors: Using several variables to predict the outcome
- Coefficient interpretation: Effect of each predictor while holding others constant
- Adjusted R-squared: R-squared adjusted for the number of predictors
- Feature importance: Determining which predictors have the strongest effect
            """)
            
            st.code('''# Perform multiple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Prepare data for multiple regression
data_copy = data.copy()

# Convert potential numeric columns to numeric
for col in data_copy.columns:
    if col not in ["Deaths", "Population"]:
        try:
            data_copy[col] = pd.to_numeric(data_copy[col], errors="coerce")
        except:
            pass

# Let's use both Population and any additional numeric features
numeric_cols = ["Deaths", "Population"]
for col in data_copy.columns:
    if col not in numeric_cols and pd.api.types.is_numeric_dtype(data_copy[col]):
        if not data_copy[col].isna().all() and len(data_copy[col].unique()) > 1:
            numeric_cols.append(col)

# Check if we have enough numeric columns for multiple regression
if len(numeric_cols) <= 2:
    print("Not enough numeric predictors for multiple regression.")
    print("Using dummy data to demonstrate the concept...")
    
    # Create dummy data if needed
    state_summary = data_copy.groupby("State").agg({
        "Deaths": "sum",
        "Population": "mean"
    }).reset_index()
    
    # Generate a dummy 'Age_Group' feature for demonstration
    np.random.seed(42)
    state_summary["Avg_Age"] = np.random.uniform(35, 50, size=len(state_summary))
    state_summary["Poverty_Rate"] = np.random.uniform(5, 25, size=len(state_summary))
    
    # Define predictors and outcome
    X_cols = ["Population", "Avg_Age", "Poverty_Rate"]
else:
    # Group data by state and calculate aggregate statistics
    agg_dict = {"Deaths": "sum", "Population": "mean"}
    for col in numeric_cols:
        if col not in ["Deaths", "State"]:
            agg_dict[col] = "mean"
    
    state_summary = data_copy.groupby("State").agg(agg_dict).reset_index()
    
    # Define predictors (excluding 'Deaths' and 'State')
    X_cols = [col for col in numeric_cols if col not in ["Deaths", "State"]]

# Clean up data and prepare for regression
state_summary = state_summary.dropna(subset=["Deaths"] + X_cols)

# Define predictors and outcome
X = state_summary[X_cols].values
y = state_summary["Deaths"].values

# Scale the features to have mean=0 and variance=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit the model
model = LinearRegression()
model.fit(X_scaled, y)

# Get model coefficients and metrics
coefficients = model.coef_
intercept = model.intercept_
r_squared = model.score(X_scaled, y)
adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

# Make predictions
y_pred = model.predict(X_scaled)

# Calculate error metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# Display results
print("Multiple Linear Regression Results:")
print(f"Intercept: {intercept:.2f}")
print("Coefficients:")
for i, col in enumerate(X_cols):
    print(f"  {col}: {coefficients[i]:.6f}")

print(f"\nR-squared: {r_squared:.4f}")
print(f"Adjusted R-squared: {adjusted_r_squared:.4f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Calculate and visualize feature importance
importance = np.abs(coefficients)
importance = 100.0 * (importance / np.sum(importance))

# Create a bar chart of feature importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({"Feature": X_cols, "Importance": importance})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("Feature Importance in Multiple Regression Model")
plt.xlabel("Relative Importance (%)")
plt.tight_layout()
plt.show()

# Create a scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.title("Actual vs. Predicted Death Counts")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()''', language="python")
            
            # Execute the code with the real data
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for multiple regression
            data_copy = data.copy()
            
            # Convert potential numeric columns to numeric
            for col in data_copy.columns:
                if col not in ["Deaths", "Population"]:
                    try:
                        data_copy[col] = pd.to_numeric(data_copy[col], errors="coerce")
                    except:
                        pass
            
            # Let's use both Population and any additional numeric features
            numeric_cols = ["Deaths", "Population"]
            for col in data_copy.columns:
                if col not in numeric_cols and pd.api.types.is_numeric_dtype(data_copy[col]):
                    if not data_copy[col].isna().all() and len(data_copy[col].unique()) > 1:
                        numeric_cols.append(col)
            
            # Check if we have enough numeric columns for multiple regression
            if len(numeric_cols) <= 2:
                st.info("Not enough numeric predictors for multiple regression in the actual dataset.")
                st.write("Using synthesized features to demonstrate the concept...")
                
                # Create demo data
                state_summary = data_copy.groupby("State").agg({
                    "Deaths": "sum",
                    "Population": "mean"
                }).reset_index()
                
                # Generate dummy features for demonstration
                np.random.seed(42)
                state_summary["Avg_Age"] = np.random.uniform(35, 50, size=len(state_summary))
                state_summary["Poverty_Rate"] = np.random.uniform(5, 25, size=len(state_summary))
                
                # Define predictors and outcome
                X_cols = ["Population", "Avg_Age", "Poverty_Rate"]
            else:
                # Group data by state and calculate aggregate statistics
                agg_dict = {"Deaths": "sum", "Population": "mean"}
                for col in numeric_cols:
                    if col not in ["Deaths", "State"]:
                        agg_dict[col] = "mean"
                
                state_summary = data_copy.groupby("State").agg(agg_dict).reset_index()
                
                # Define predictors (excluding 'Deaths' and 'State')
                X_cols = [col for col in numeric_cols if col not in ["Deaths", "State"]]
            
            # Display which variables we're using
            st.write(f"Variables used in the model: {', '.join(X_cols)}")
            
            # Clean up data and prepare for regression
            state_summary = state_summary.dropna(subset=["Deaths"] + X_cols)
            
            # Define predictors and outcome
            X = state_summary[X_cols].values
            y = state_summary["Deaths"].values
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create and fit the model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Get model coefficients and metrics
            coefficients = model.coef_
            intercept = model.intercept_
            r_squared = model.score(X_scaled, y)
            adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Calculate error metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Display results
            st.write("Multiple Linear Regression Results:")
            st.info(f"Intercept: {intercept:.2f}")
            
            st.write("Coefficients:")
            coef_df = pd.DataFrame({
                "Feature": X_cols,
                "Coefficient": coefficients
            })
            st.dataframe(coef_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R-squared", f"{r_squared:.4f}")
            with col2:
                st.metric("Adjusted R-squared", f"{adjusted_r_squared:.4f}")
            with col3:
                st.metric("RMSE", f"{rmse:.2f}")
            
            # Calculate and visualize feature importance
            importance = np.abs(coefficients)
            importance = 100.0 * (importance / np.sum(importance))
            
            # Create a bar chart of feature importance
            feature_importance = pd.DataFrame({"Feature": X_cols, "Importance": importance})
            feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
            ax.set_title("Feature Importance in Multiple Regression Model")
            ax.set_xlabel("Relative Importance (%)")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Create a scatter plot of actual vs. predicted values
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y, y_pred)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax.set_xlabel("Actual Deaths")
            ax.set_ylabel("Predicted Deaths")
            ax.set_title("Actual vs. Predicted Death Counts")
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
**Understanding Multiple Regression Results:**

1. **Interpreting Coefficients:**
   - Each coefficient represents the change in the outcome for a one-unit change in the predictor, holding all other predictors constant
   - Because we scaled the features, coefficients can be compared directly to assess relative importance
   - Positive coefficients indicate positive relationships, negative coefficients indicate inverse relationships

2. **Model Quality Metrics:**
   - R-squared: Proportion of variance explained by the model
   - Adjusted R-squared: Accounts for the number of predictors (prevents overfitting)
   - RMSE: Average prediction error in the original units of the dependent variable

3. **Feature Importance:**
   - Feature importance helps identify which predictors have the strongest effect on the outcome
   - More important features may be priorities for interventions or policy changes

4. **Actual vs. Predicted Plot:**
   - Points close to the diagonal line indicate accurate predictions
   - Systematic deviations suggest model limitations
   - Outliers represent cases where the model performs poorly
            """)
            
        with st.expander("📊 Logistic Regression for Classification"):
            st.markdown("""
### Logistic Regression for Classification

**Objective:**  
Predict binary outcomes (yes/no, 0/1) using logistic regression.

**Key Concepts:**
- Binary classification: Predicting one of two possible outcomes
- Probability estimation: Model predicts probability of the positive class
- Threshold: Cut-off probability for assigning the positive class
- Confusion matrix: Table showing true vs. predicted classifications
- Evaluation metrics: Accuracy, precision, recall, F1-score, AUC-ROC
            """)
            
            st.code('''# Perform logistic regression for classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score

# Prepare the data
data_copy = data.copy()
data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")

# Create a binary classification target based on death rate
state_summary = data_copy.groupby("State").agg({
    "Deaths": "sum",
    "Population": "mean"
}).reset_index()

# Calculate death rate per 100,000 population
state_summary["Death_Rate"] = (state_summary["Deaths"] / state_summary["Population"]) * 100000

# Create binary target: 1 if death rate is above median, 0 otherwise
median_rate = state_summary["Death_Rate"].median()
state_summary["High_Death_Rate"] = (state_summary["Death_Rate"] > median_rate).astype(int)

# If we don't have additional features, create some for demonstration
if "Cause of death Code" in data_copy.columns:
    # Create features based on cause of death categories
    def get_icd_category(code):
        if pd.isna(code):
            return "Unknown"
        first_char = str(code)[0]
        return first_char
    
    # Calculate percentage of deaths by ICD category for each state
    data_copy["ICD_Category"] = data_copy["Cause of death Code"].apply(get_icd_category)
    icd_pivots = pd.pivot_table(
        data_copy, 
        values="Deaths", 
        index="State", 
        columns="ICD_Category",
        aggfunc="sum",
        fill_value=0
    )
    
    # Convert to percentages
    icd_percentages = icd_pivots.div(icd_pivots.sum(axis=1), axis=0) * 100
    
    # Rename columns to make it clear these are percentages
    icd_percentages.columns = [f"Pct_ICD_{col}" for col in icd_percentages.columns]
    
    # Merge with state_summary
    state_summary = pd.merge(state_summary, icd_percentages.reset_index(), on="State")
    
    # Select features (population and ICD percentages)
    feature_cols = ["Population"] + [col for col in state_summary.columns if col.startswith("Pct_ICD_")]
else:
    st.info("Using synthetic features for demonstration since cause of death codes are not available.")
    # Create synthetic features for demonstration
    np.random.seed(42)
    state_summary["Pct_Elderly"] = np.random.uniform(10, 35, len(state_summary))
    state_summary["Pct_Smoking"] = np.random.uniform(15, 30, len(state_summary))
    state_summary["Pct_Obesity"] = np.random.uniform(20, 40, len(state_summary))
    
    # Select features
    feature_cols = ["Population", "Pct_Elderly", "Pct_Smoking", "Pct_Obesity"]

# Remove any rows with missing values
state_summary = state_summary.dropna(subset=feature_cols + ["High_Death_Rate"])

# Display data shape after cleaning
st.write(f"Data after cleaning: {state_summary.shape[0]} states with {len(feature_cols)} features")

# Check if we have enough data to split
if len(state_summary) < 10 or state_summary.shape[0] == 0:
    st.error("Not enough data for meaningful train/test split. Need at least 10 states with complete data.")
    st.warning("Showing dummy results for educational purposes only.")
    st.error("Not enough data for meaningful train/test split. Need at least 10 states with complete data.")
    st.warning("Showing dummy results for educational purposes only.")
    st.error("Not enough data for meaningful train/test split. Need at least 10 states with complete data.")
    st.warning("Showing dummy results for educational purposes only.")
    
    # Create dummy data for demonstration
    np.random.seed(42)
    n_samples = 50
    X_dummy = np.random.randn(n_samples, 4)
    y_dummy = np.random.randint(0, 2, n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.3, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create dummy feature importance
    coefficients = model.coef_[0]
    dummy_features = ["Population", "Feature_2", "Feature_3", "Feature_4"]
    
    st.warning("Using dummy data - results are for illustration only!")
else:
    # Split data into features and target
    X = state_summary[feature_cols].values
    y = state_summary["High_Death_Rate"].values
    
    # Additional check to ensure we have data after selecting features
    if X.shape[0] == 0:
        st.error("No valid data after feature selection. Showing dummy results instead.")
        # Create dummy data for demonstration
        np.random.seed(42)
        n_samples = 50
        X_dummy = np.random.randn(n_samples, 4)
        y_dummy = np.random.randint(0, 2, n_samples)
        
        X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.3, random_state=42)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create dummy feature importance
        coefficients = model.coef_[0]
        dummy_features = ["Population", "Feature_2", "Feature_3", "Feature_4"]
        
        st.warning("Using dummy data - results are for illustration only!")
    else:
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        st.write(f"Training set: {X_train.shape[0]} states • Test set: {X_test.shape[0]} states")
        
        # Create and fit the logistic regression model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of high death rate
        y_pred = model.predict(X_test)  # Binary prediction using default threshold (0.5)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Store feature names for importance plots
        coefficients = model.coef_[0]

# Display results
st.write("### Logistic Regression Results for Predicting High Death Rate")

col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{accuracy:.4f}")
with col2:
    st.metric("AUC-ROC", f"{roc_auc:.4f}")

st.write("#### Confusion Matrix")
# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Rate", "High Rate"],
            yticklabels=["Low Rate", "High Rate"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
st.pyplot(fig)

st.write("#### Classification Report")
# Get classification report as a string
from io import StringIO
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Get feature importance from model coefficients
importance = np.abs(coefficients)
importance = 100.0 * (importance / np.sum(importance))

# Create a DataFrame for feature importance
if 'dummy_features' in locals():
    feature_list = dummy_features
else:
    feature_list = feature_cols

feature_importance = pd.DataFrame({
    "Feature": feature_list,
    "Coefficient": coefficients,
    "Importance": importance
})
feature_importance = feature_importance.sort_values("Importance", ascending=False)

st.write("#### Feature Importance")
st.dataframe(feature_importance)

# Visualize ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
ax.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
st.pyplot(fig)

# Visualize feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
ax.set_title("Feature Importance in Logistic Regression Model")
ax.set_xlabel("Relative Importance (%)")
plt.tight_layout()
st.pyplot(fig)''', language="python")
            
            # Execute the code with the real data
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
            
            # Prepare the data
            data_copy = data.copy()
            data_copy["Deaths"] = pd.to_numeric(data_copy["Deaths"], errors="coerce")
            data_copy["Population"] = pd.to_numeric(data_copy["Population"], errors="coerce")
            
            # Create a binary classification target based on death rate
            state_summary = data_copy.groupby("State").agg({
                "Deaths": "sum",
                "Population": "mean"
            }).reset_index()
            
            # Calculate death rate per 100,000 population
            state_summary["Death_Rate"] = (state_summary["Deaths"] / state_summary["Population"]) * 100000
            
            # Create binary target: 1 if death rate is above median, 0 otherwise
            median_rate = state_summary["Death_Rate"].median()
            state_summary["High_Death_Rate"] = (state_summary["Death_Rate"] > median_rate).astype(int)
            
            st.write("Created target variable: States with above-median death rates (1) vs. below-median (0)")
            # Display states and their classification
            high_rate_states = state_summary[state_summary["High_Death_Rate"] == 1]["State"].tolist()
            low_rate_states = state_summary[state_summary["High_Death_Rate"] == 0]["State"].tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**High Death Rate States:**")
                st.write(", ".join(high_rate_states[:10]) + ("..." if len(high_rate_states) > 10 else ""))
            with col2:
                st.write("**Low Death Rate States:**")
                st.write(", ".join(low_rate_states[:10]) + ("..." if len(low_rate_states) > 10 else ""))
            
            # Check if we have cause of death data to create features
            if "Cause of death Code" in data_copy.columns:
                # Create features based on cause of death categories
                def get_icd_category(code):
                    if pd.isna(code):
                        return "Unknown"
                    first_char = str(code)[0]
                    return first_char
                
                # Calculate percentage of deaths by ICD category for each state
                data_copy["ICD_Category"] = data_copy["Cause of death Code"].apply(get_icd_category)
                try:
                    icd_pivots = pd.pivot_table(
                        data_copy, 
                        values="Deaths", 
                        index="State", 
                        columns="ICD_Category",
                        aggfunc="sum",
                        fill_value=0
                    )
                    
                    # Convert to percentages
                    icd_percentages = icd_pivots.div(icd_pivots.sum(axis=1), axis=0) * 100
                    
                    # Rename columns to make it clear these are percentages
                    icd_percentages.columns = [f"Pct_ICD_{col}" for col in icd_percentages.columns]
                    
                    # Merge with state_summary
                    state_summary = pd.merge(state_summary, icd_percentages.reset_index(), on="State")
                    
                    # Select features (population and ICD percentages)
                    feature_cols = ["Population"] + [col for col in state_summary.columns if col.startswith("Pct_ICD_")]
                except Exception as e:
                    st.warning(f"Error creating ICD features: {e}. Using synthetic features instead.")
                    # Create synthetic features for demonstration
                    np.random.seed(42)
                    state_summary["Pct_Elderly"] = np.random.uniform(10, 35, len(state_summary))
                    state_summary["Pct_Smoking"] = np.random.uniform(15, 30, len(state_summary))
                    state_summary["Pct_Obesity"] = np.random.uniform(20, 40, len(state_summary))
                    
                    # Select features
                    feature_cols = ["Population", "Pct_Elderly", "Pct_Smoking", "Pct_Obesity"]



































            
            
            
            
            
            
            
elif st.session_state.page == "template":
    st.header("🧩 Create Custom Template")
    if not st.session_state.template_created:
        st.markdown("### 1. Select Sections for Your Template")
        st.markdown("Choose the topic sections to include in your custom cheatsheet:")
        sections_col1, sections_col2 = st.columns(2)
        with sections_col1:
            for section in list(template_content.keys())[:2]:
                st.session_state.template_selections[section]["selected"] = st.checkbox(
                    f"{section}", 
                    value=st.session_state.template_selections[section]["selected"],
                    key=f"section_{section}"
                )
                if st.session_state.template_selections[section]["selected"]:
                    st.markdown("Select difficulty levels:")
                    for level in ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"]:
                        st.session_state.template_selections[section]["levels"][level] = st.checkbox(
                            f"  {level}", 
                            value=st.session_state.template_selections[section]["levels"][level],
                            key=f"level_{section}_{level}"
                        )
                        if st.session_state.template_selections[section]["levels"][level]:
                            st.markdown("Select topics:")
                            for subsection in template_content[section][level].keys():
                                if subsection not in st.session_state.template_selections[section]["subsections"]:
                                    st.session_state.template_selections[section]["subsections"][subsection] = False
                                st.session_state.template_selections[section]["subsections"][subsection] = st.checkbox(
                                    f"    {subsection}", 
                                    value=st.session_state.template_selections[section]["subsections"].get(subsection, False),
                                    key=f"subsection_{section}_{level}_{subsection}"
                                )
                st.markdown("---")
        with sections_col2:
            for section in list(template_content.keys())[2:]:
                st.session_state.template_selections[section]["selected"] = st.checkbox(
                    f"{section}", 
                    value=st.session_state.template_selections[section]["selected"],
                    key=f"section_{section}"
                )
                if st.session_state.template_selections[section]["selected"]:
                    st.markdown("Select difficulty levels:")
                    for level in ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"]:
                        st.session_state.template_selections[section]["levels"][level] = st.checkbox(
                            f"  {level}", 
                            value=st.session_state.template_selections[section]["levels"][level],
                            key=f"level_{section}_{level}"
                        )
                        if st.session_state.template_selections[section]["levels"][level]:
                            st.markdown("Select topics:")
                            for subsection in template_content[section][level].keys():
                                if subsection not in st.session_state.template_selections[section]["subsections"]:
                                    st.session_state.template_selections[section]["subsections"][subsection] = False
                                st.session_state.template_selections[section]["subsections"][subsection] = st.checkbox(
                                    f"    {subsection}", 
                                    value=st.session_state.template_selections[section]["subsections"].get(subsection, False),
                                    key=f"subsection_{section}_{level}_{subsection}"
                                )
                st.markdown("---")
        st.markdown("### Template Summary")
        total_sections = sum(1 for section, data in st.session_state.template_selections.items() if data["selected"])
        total_subsections = 0
        for section, data in st.session_state.template_selections.items():
            if data["selected"]:
                for subsection, selected in data["subsections"].items():
                    if selected:
                        total_subsections += 1
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sections Selected", total_sections)
        with col2:
            st.metric("Topics Selected", total_subsections)
        with col3:
            selected_levels = set()
            for section, data in st.session_state.template_selections.items():
                if data["selected"]:
                    for level, selected in data["levels"].items():
                        if selected:
                            selected_levels.add(level)
            st.metric("Difficulty Levels", len(selected_levels))
        st.markdown("#### Selected Sections")
        for section, data in st.session_state.template_selections.items():
            if data["selected"]:
                section_total = sum(1 for sub, sel in data["subsections"].items() if sel)
                section_possible = 0
                for level in ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"]:
                    if data["levels"][level]:
                        section_possible += len(template_content[section][level])
                percentage = int((section_total / section_possible) * 100) if section_possible > 0 else 0
                st.markdown(f"**{section}**: {section_total} topics selected")
                st.progress(percentage / 100)
        st.markdown("### 2. Name Your Template")
        template_name = st.text_input("Template Name:", value="My Custom Cheatsheet")
        if st.button("Generate Template", use_container_width=True):
            st.session_state.template_name = template_name
            st.session_state.template_created = True
            st.rerun()
    else:
        st.success(f"✅ Your template '{st.session_state.template_name}' has been created!")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("← Back to Knowledge Graph"):
                st.session_state.page = "graph"
                st.session_state.template_created = False
                st.rerun()
        with col2:
            if st.button("🔄 Create Another Template"):
                st.session_state.template_created = False
                st.rerun()
        with col3:
            template_md = ""
            for section, section_data in st.session_state.template_selections.items():
                if section_data["selected"]:
                    template_md += f"# {section}\n\n"
                    for level in ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"]:
                        if section_data["levels"][level]:
                            template_md += f"## {level}\n\n"
                            for subsection, selected in section_data["subsections"].items():
                                if selected and subsection in template_content[section][level]:
                                    template_md += f"### {subsection}\n\n"
                                    template_md += f"{template_content[section][level][subsection]['description']}\n\n"
                                    template_md += f"```python\n{template_content[section][level][subsection]['code']}\n```\n\n"
                                    template_md += f"Example:\n```python\n{template_content[section][level][subsection]['example']}\n```\n\n"
                                    if 'output' in template_content[section][level][subsection]:
                                        template_md += f"Output:\n```\n{template_content[section][level][subsection]['output']}\n```\n\n---\n\n"
                                    else:
                                        template_md += "---\n\n"
            st.download_button(
                label="📥 Download Template",
                data=template_md,
                file_name=f"{st.session_state.template_name.replace(' ', '_')}.md",
                mime="text/markdown"
            )
        st.markdown("---")
        st.markdown(f"## 📚 {st.session_state.template_name}")
        has_content = False
        for section, section_data in st.session_state.template_selections.items():
            if section_data["selected"]:
                st.markdown(f"## {section}")
                for level in ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"]:
                    if section_data["levels"][level]:
                        level_content = []
                        for subsection, selected in section_data["subsections"].items():
                            if selected and subsection in template_content[section][level]:
                                has_content = True
                                level_content.append(subsection)
                        if level_content:
                            st.markdown(f"### {level}")
                            for subsection in level_content:
                                with st.expander(f"**{subsection}**"):
                                    st.markdown(f"**Description:** {template_content[section][level][subsection]['description']}")
                                    st.markdown("**Code:**")
                                    st.code(template_content[section][level][subsection]['code'], language="python")
                                    st.markdown("**Example:**")
                                    st.code(template_content[section][level][subsection]['example'], language="python")
                                    if 'output' in template_content[section][level][subsection]:
                                        st.markdown("**Output:**")
                                        st.code(template_content[section][level][subsection]['output'])
                                    if section == "Load Data" and "read_" in subsection.lower():
                                        st.markdown("**Sample Data Preview:**")
                                        st.dataframe(data.head())
        if not has_content:
            st.warning("No content selected. Please go back and select at least one section, level, and topic.")
        if any(st.session_state.template_selections.get("Load Data", {}).get("subsections", {}).values()):
            st.markdown("### 📊 Sample Dataset Preview")
            st.markdown("This is the dataset used in the code templates above:")
            st.dataframe(data.head())
else:
    st.info("Please navigate to a page via the sidebar navigation.")
