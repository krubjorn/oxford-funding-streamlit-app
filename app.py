import pathlib
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Oxford Funding Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = pathlib.Path(__file__).parent
PROJECT_DATA_PATH = APP_DIR / "data" / "projectsearch.csv"
SUCCESS_DATA_PATH = APP_DIR / "data" / "success_rate_data.csv"

DEFAULT_TARGET_DEPARTMENTS = [
    "Oxford Martin School",
    "Social Sciences Division",
    "Smith School of Enterprise and the Env",
    "Geography - SoGE",
    "Environmental Change Institute SoGE",
    "Transport Studies Unit SoGE",
]

DEFAULT_SELECTED_FUNDERS = [
    "European Commission",
    "European Space Agency",
    "Department of Environment Food and Rural Affairs",
    "Department for Transport",
    "Economic & Social Research Council",
    "Leverhulme Trust",
    "Natural Environment Research Council",
    "National Academy of Sciences",
    "Met Office",
    "Open Society Foundations",
    "Royal Society",
    "Science and Technology Facilities Council",
    "UK Research and Innovation (UKRI)",
    "UK Research and Innovation/ European Research Council",
    "Wellcome Trust",
]

REQUIRED_PROJECT_COLUMNS = {
    "LeadROName",
    "FundingOrgName",
    "Department",
    "AwardPounds",
    "ProjectId",
}

REQUIRED_SUCCESS_COLUMNS = {
    "Financial Year",
    "Fund Deci Status Desc",
    "Funder",
    "Project Ref",
    "Scheme Name",
}


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1.1rem;
        background: linear-gradient(135deg, #102a43 0%, #243b53 45%, #486581 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 12px 34px rgba(16, 42, 67, 0.18);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        line-height: 1.15;
    }
    .hero p {
        margin: 0.65rem 0 0 0;
        color: #d9e2ec;
        font-size: 1rem;
    }
    .section-note {
        padding: 0.8rem 1rem;
        border-left: 4px solid #486581;
        background: #f0f4f8;
        border-radius: 0.6rem;
        color: #243b53;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    }
    .small-caption {
        font-size: 0.85rem;
        color: #52616b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def currency_gbp(value: float) -> str:
    if pd.isna(value):
        return "£0"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"£{value / 1_000_000_000:,.2f}bn"
    if abs_value >= 1_000_000:
        return f"£{value / 1_000_000:,.2f}m"
    if abs_value >= 1_000:
        return f"£{value / 1_000:,.1f}k"
    return f"£{value:,.0f}"


def clean_money(series: pd.Series) -> pd.Series:
    """Convert currency-like strings such as '£1,200' into numbers."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("£", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def extract_year(value) -> float:
    """Extract a calendar-like year from values such as '2020/21' or '20/21'."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    four_digit = pd.Series([text]).str.extract(r"(20\d{2})", expand=False).iloc[0]
    if pd.notna(four_digit):
        return int(four_digit)
    digits = "".join(c for c in text if c.isdigit())
    if len(digits) >= 2:
        return int("20" + digits[:2])
    return np.nan


def missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required) - set(df.columns))


@st.cache_data(show_spinner=False)
def load_csv_from_repo(path: str) -> pd.DataFrame | None:
    file_path = pathlib.Path(path)
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def prepare_project_data(raw_df: pd.DataFrame, lead_ro_name: str) -> pd.DataFrame:
    df = raw_df.copy()

    if "AwardPounds" in df.columns:
        df["AwardPounds"] = clean_money(df["AwardPounds"])

    if "StartYear" not in df.columns:
        if "StartDate" in df.columns:
            df["StartDate"] = pd.to_datetime(df["StartDate"], dayfirst=True, errors="coerce")
            df["StartYear"] = df["StartDate"].dt.year
        else:
            df["StartYear"] = np.nan
    else:
        df["StartYear"] = pd.to_numeric(df["StartYear"], errors="coerce")

    if "LeadROName" in df.columns and lead_ro_name:
        df = df[df["LeadROName"].astype(str).eq(lead_ro_name)].copy()

    df = df.dropna(subset=["StartYear"])
    df["StartYear"] = df["StartYear"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def prepare_success_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["Year"] = df["Financial Year"].apply(extract_year)

    status_map = {
        "Approved by funder": "A",
        "Rejected by funder": "U",
        "Submitted to funder": "S",
        "A": "A",
        "U": "U",
        "S": "S",
    }
    df["Fund Deci Status Desc"] = (
        df["Fund Deci Status Desc"].map(status_map).fillna(df["Fund Deci Status Desc"])
    )
    return df


def apply_project_filters(
    df: pd.DataFrame,
    departments: list[str],
    funders: list[str],
    year_range: tuple[int, int] | None,
) -> pd.DataFrame:
    filtered = df.copy()
    if departments:
        filtered = filtered[filtered["Department"].isin(departments)]
    if funders:
        filtered = filtered[filtered["FundingOrgName"].isin(funders)]
    if year_range is not None:
        filtered = filtered[
            filtered["StartYear"].between(int(year_range[0]), int(year_range[1]))
        ]
    return filtered


def apply_success_filters(
    df: pd.DataFrame,
    funders: list[str],
    year_range: tuple[int, int] | None,
) -> pd.DataFrame:
    filtered = df.copy()
    if funders:
        filtered = filtered[filtered["Funder"].isin(funders)]
    if year_range is not None and "Year" in filtered.columns:
        filtered = filtered[filtered["Year"].between(int(year_range[0]), int(year_range[1]))]
    return filtered


def standard_layout(fig, height: int = 600):
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=30, r=30, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
    )
    return fig


def show_project_metrics(df: pd.DataFrame) -> None:
    total_awards = df["ProjectId"].nunique() if "ProjectId" in df else len(df)
    total_funding = df["AwardPounds"].sum(skipna=True) if "AwardPounds" in df else 0
    year_min = int(df["StartYear"].min()) if not df.empty else "-"
    year_max = int(df["StartYear"].max()) if not df.empty else "-"
    departments = df["Department"].nunique() if "Department" in df else 0
    funders = df["FundingOrgName"].nunique() if "FundingOrgName" in df else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Awards", f"{total_awards:,}")
    c2.metric("Total funding", currency_gbp(total_funding))
    c3.metric("Year range", f"{year_min}–{year_max}")
    c4.metric("Departments / funders", f"{departments:,} / {funders:,}")


def show_success_metrics(df: pd.DataFrame) -> None:
    submitted = len(df)
    successful = (df["Fund Deci Status Desc"] == "A").sum()
    success_rate = 100 * successful / submitted if submitted else 0
    funders = df["Funder"].nunique() if "Funder" in df else 0
    years = df["Year"].dropna().nunique() if "Year" in df else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{submitted:,}")
    c2.metric("Successful", f"{successful:,}")
    c3.metric("Success rate", f"{success_rate:.1f}%")
    c4.metric("Funders / years", f"{funders:,} / {years:,}")


# -----------------------------------------------------------------------------
# Plot builders: project funding
# -----------------------------------------------------------------------------
def plot_overall_awards_by_year(df: pd.DataFrame):
    yearly_summary = (
        df.groupby("StartYear")
        .agg(
            number_of_awards=("ProjectId", "nunique"),
            total_award_pounds=("AwardPounds", "sum"),
        )
        .reset_index()
    )
    fig = px.scatter(
        yearly_summary,
        x="StartYear",
        y="number_of_awards",
        size="total_award_pounds",
        hover_name="StartYear",
        hover_data={
            "total_award_pounds": ":,.2f",
            "number_of_awards": True,
            "StartYear": False,
        },
        title="Number of Awards and Total Funding by Year",
        labels={
            "StartYear": "Year",
            "number_of_awards": "Number of awards",
            "total_award_pounds": "Total award pounds",
        },
        size_max=70,
    )
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
    return standard_layout(fig)


def plot_department_yearly_awards(df: pd.DataFrame, department: str):
    dept_df = df[df["Department"].eq(department)].copy()
    yearly = dept_df.groupby("StartYear", as_index=False)["AwardPounds"].sum()
    fig = px.bar(
        yearly,
        x="StartYear",
        y="AwardPounds",
        title=f"Total Award Pounds by Year: {department}",
        labels={"StartYear": "Year", "AwardPounds": "Funding awarded (£)"},
        hover_data={"StartYear": True, "AwardPounds": ":,.2f"},
    )
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
    return standard_layout(fig)


def plot_selected_departments_bubble(df: pd.DataFrame, include_funder: bool):
    group_cols = ["StartYear", "Department"] + (["FundingOrgName"] if include_funder else [])
    summary = (
        df.groupby(group_cols)
        .agg(
            number_of_awards=("ProjectId", "nunique"),
            total_award_pounds=("AwardPounds", "sum"),
        )
        .reset_index()
    )
    hover_data = {
        "StartYear": True,
        "total_award_pounds": ":,.2f",
        "number_of_awards": True,
    }
    if include_funder:
        hover_data["FundingOrgName"] = True

    fig = px.scatter(
        summary,
        x="StartYear",
        y="number_of_awards",
        size="total_award_pounds",
        color="Department",
        hover_name="Department",
        hover_data=hover_data,
        title="Awards and Funding by Year for Selected Departments",
        labels={
            "StartYear": "Year",
            "number_of_awards": "Number of awards",
            "total_award_pounds": "Total award pounds",
            "FundingOrgName": "Funding organisation",
        },
        size_max=65,
    )
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
    return standard_layout(fig)


def plot_funding_org_bubble(df: pd.DataFrame, color_by_department: bool):
    group_cols = ["StartYear", "FundingOrgName"] + (["Department"] if color_by_department else [])
    summary = (
        df.groupby(group_cols)
        .agg(
            number_of_awards=("ProjectId", "nunique"),
            total_award_pounds=("AwardPounds", "sum"),
        )
        .reset_index()
    )
    color = "Department" if color_by_department else "FundingOrgName"
    hover_data = {
        "StartYear": True,
        "total_award_pounds": ":,.2f",
        "number_of_awards": True,
    }
    if "Department" in summary.columns:
        hover_data["Department"] = True

    fig = px.scatter(
        summary,
        x="StartYear",
        y="number_of_awards",
        size="total_award_pounds",
        color=color,
        hover_name="FundingOrgName",
        hover_data=hover_data,
        title="Number of Awards, Funding by Year and Funding Organisation",
        labels={
            "StartYear": "Year",
            "number_of_awards": "Number of awards",
            "total_award_pounds": "Total award pounds",
            "FundingOrgName": "Funding organisation",
            "Department": "Department",
        },
        size_max=65,
    )
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
    return standard_layout(fig)


def top_funding_org_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["FundingOrgName", "Department"], as_index=False)["AwardPounds"]
        .sum()
        .sort_values("AwardPounds", ascending=False)
    )


def average_awards_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["FundingOrgName", "Department"], as_index=False)["AwardPounds"]
        .mean()
        .sort_values("AwardPounds", ascending=False)
    )


def plot_recent_year_pies(df: pd.DataFrame, max_years: int = 6):
    years = sorted(df["StartYear"].dropna().astype(int).unique(), reverse=True)[:max_years]
    for i in range(0, len(years), 2):
        cols = st.columns(2)
        for col, year in zip(cols, years[i : i + 2]):
            df_year = df[df["StartYear"].eq(year)]
            funding_org_awards = (
                df_year.groupby("FundingOrgName")
                .agg(
                    number_of_awards=("ProjectId", "nunique"),
                    departments_involved=(
                        "Department",
                        lambda x: ", ".join(sorted(x.dropna().astype(str).unique())),
                    ),
                )
                .reset_index()
                .sort_values("number_of_awards", ascending=False)
            )
            with col:
                if funding_org_awards.empty:
                    st.info(f"No data for {year}.")
                    continue
                fig = px.pie(
                    funding_org_awards,
                    values="number_of_awards",
                    names="FundingOrgName",
                    title=f"Awards by funding organisation in {year}",
                    hole=0.45,
                    hover_data=["number_of_awards", "departments_involved"],
                    labels={
                        "FundingOrgName": "Funding organisation",
                        "number_of_awards": "Number of awards",
                        "departments_involved": "Departments",
                    },
                )
                fig.update_traces(textposition="inside", textinfo="percent")
                st.plotly_chart(standard_layout(fig, height=480), use_container_width=True)


# -----------------------------------------------------------------------------
# Plot builders: success-rate assessment
# -----------------------------------------------------------------------------
def get_funder_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Funder")
        .agg(
            Submitted=("Project Ref", "count"),
            Successful=("Fund Deci Status Desc", lambda x: (x == "A").sum()),
        )
        .reset_index()
    )
    summary["Success Rate"] = np.where(
        summary["Submitted"] > 0,
        100 * summary["Successful"] / summary["Submitted"],
        0,
    )
    return summary.sort_values("Success Rate", ascending=False)


def plot_success_by_funder(df: pd.DataFrame):
    summary = get_funder_summary(df)
    fig = px.bar(
        summary,
        x="Success Rate",
        y="Funder",
        orientation="h",
        color="Success Rate",
        color_continuous_scale="RdYlGn",
        hover_data=["Submitted", "Successful"],
        title="Success Rate by Funder",
        labels={"Success Rate": "Success rate (%)"},
    )
    height = max(650, len(summary) * 30)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return standard_layout(fig, height=height)


def plot_success_heatmap(df: pd.DataFrame, top_n_funders: int):
    funder_year_summary = (
        df.groupby(["Funder", "Year"])
        .agg(
            Submitted=("Project Ref", "count"),
            Successful=("Fund Deci Status Desc", lambda x: (x == "A").sum()),
        )
        .reset_index()
    )
    funder_year_summary["Success Rate"] = np.where(
        funder_year_summary["Submitted"] > 0,
        100 * funder_year_summary["Successful"] / funder_year_summary["Submitted"],
        np.nan,
    )
    top_funders = (
        funder_year_summary.groupby("Funder")["Submitted"]
        .sum()
        .nlargest(top_n_funders)
        .index.tolist()
    )
    filtered = funder_year_summary[funder_year_summary["Funder"].isin(top_funders)]
    heatmap_data = filtered.pivot(index="Funder", columns="Year", values="Success Rate")

    fig = px.imshow(
        heatmap_data,
        aspect="auto",
        text_auto=".0f",
        color_continuous_scale="RdYlGn",
        title=f"Success Rate by Funder and Year: Top {top_n_funders} Funders",
        labels=dict(x="Year", y="Funder", color="Success rate (%)"),
    )
    fig.update_layout(
        height=max(700, len(heatmap_data) * 42),
        margin=dict(l=330, r=40, t=80, b=50),
        template="plotly_white",
    )
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    return fig


def get_scheme_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Funder", "Scheme Name"])
        .agg(
            Submitted=("Project Ref", "count"),
            Successful=("Fund Deci Status Desc", lambda x: (x == "A").sum()),
        )
        .reset_index()
    )
    summary["Success Rate"] = np.where(
        summary["Submitted"] > 0,
        100 * summary["Successful"] / summary["Submitted"],
        0,
    )
    return summary


def plot_scheme_portfolio(df: pd.DataFrame):
    summary = get_scheme_summary(df)
    if summary.empty:
        return None
    fig = px.scatter(
        summary,
        x="Submitted",
        y="Success Rate",
        size="Successful",
        color="Funder",
        hover_name="Scheme Name",
        title="Scheme Portfolio Performance",
        labels={"Success Rate": "Success rate (%)"},
        size_max=55,
    )
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=30, r=120, t=80, b=50),
    )
    return standard_layout(fig)


# -----------------------------------------------------------------------------
# Sidebar and data loading
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data inputs")
    st.markdown(
        "Upload CSV files here, or place them in the repository under `data/` with the names shown below."
    )
    project_upload = st.file_uploader(
        "Project funding CSV",
        type=["csv"],
        help="Expected default filename: data/projectsearch.csv",
    )
    success_upload = st.file_uploader(
        "Success-rate CSV",
        type=["csv"],
        help="Expected default filename: data/success_rate_data.csv",
    )
    st.divider()
    lead_ro_name = st.text_input("Lead research organisation", "University of Oxford")

raw_project = load_csv_from_upload(project_upload)
if raw_project is None:
    raw_project = load_csv_from_repo(str(PROJECT_DATA_PATH))

raw_success = load_csv_from_upload(success_upload)
if raw_success is None:
    raw_success = load_csv_from_repo(str(SUCCESS_DATA_PATH))

project_df = None
success_df = None
project_error = None
success_error = None

if raw_project is not None:
    missing = missing_columns(raw_project, REQUIRED_PROJECT_COLUMNS)
    if missing:
        project_error = f"Project CSV is missing required columns: {', '.join(missing)}"
    else:
        project_df = prepare_project_data(raw_project, lead_ro_name)

if raw_success is not None:
    missing = missing_columns(raw_success, REQUIRED_SUCCESS_COLUMNS)
    if missing:
        success_error = f"Success-rate CSV is missing required columns: {', '.join(missing)}"
    else:
        success_df = prepare_success_data(raw_success)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Oxford Funding and Application Success Dashboard</h1>
        <p>Interactive Plotly visualisations for award funding, funders, departments, years, and application success rates.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if raw_project is None and raw_success is None:
    st.info(
        "Upload your CSV files in the sidebar, or commit them to `data/projectsearch.csv` and `data/success_rate_data.csv` in your GitHub repository."
    )

if project_error:
    st.error(project_error)
if success_error:
    st.error(success_error)


# -----------------------------------------------------------------------------
# Global filters after data load
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    if project_df is not None and not project_df.empty:
        all_departments = sorted(project_df["Department"].dropna().astype(str).unique())
        default_departments = [d for d in DEFAULT_TARGET_DEPARTMENTS if d in all_departments]
        selected_departments = st.multiselect(
            "Departments for funding plots",
            all_departments,
            default=default_departments or all_departments[: min(6, len(all_departments))],
        )
        all_project_funders = sorted(project_df["FundingOrgName"].dropna().astype(str).unique())
        selected_project_funders = st.multiselect(
            "Funding organisations for project plots",
            all_project_funders,
            default=[],
            help="Leave blank to include all funding organisations.",
        )
        min_year, max_year = int(project_df["StartYear"].min()), int(project_df["StartYear"].max())
        project_year_range = st.slider(
            "Project start-year range",
            min_year,
            max_year,
            (min_year, max_year),
        )
    else:
        selected_departments = []
        selected_project_funders = []
        project_year_range = None

    if success_df is not None and not success_df.empty:
        all_success_funders = sorted(success_df["Funder"].dropna().astype(str).unique())
        default_success_funders = [f for f in DEFAULT_SELECTED_FUNDERS if f in all_success_funders]
        selected_success_funders = st.multiselect(
            "Funders for success plots",
            all_success_funders,
            default=default_success_funders,
            help="Clear the selection to include all funders.",
        )
        valid_years = success_df["Year"].dropna().astype(int)
        if valid_years.empty:
            success_year_range = None
        else:
            min_success_year, max_success_year = int(valid_years.min()), int(valid_years.max())
            success_year_range = st.slider(
                "Success-data year range",
                min_success_year,
                max_success_year,
                (min_success_year, max_success_year),
            )
        top_n_funders = st.slider("Top N funders in heatmap", 5, 30, 15)
    else:
        selected_success_funders = []
        success_year_range = None
        top_n_funders = 15


# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------
funding_tab, success_tab, about_tab = st.tabs(
    ["Funding awards", "Application success", "About / deployment"]
)

with funding_tab:
    if project_df is None or project_df.empty:
        st.warning("No project funding data is available for the selected lead research organisation.")
    else:
        project_filtered = apply_project_filters(
            project_df,
            selected_departments,
            selected_project_funders,
            project_year_range,
        )
        if project_filtered.empty:
            st.warning("No project records match the current filters.")
        else:
            show_project_metrics(project_filtered)
            st.markdown(
                "<div class='section-note'>Use the sidebar to focus on departments, funders, and year ranges. Bubble size represents total award value.</div>",
                unsafe_allow_html=True,
            )

            f1, f2, f3, f4, f5 = st.tabs(
                [
                    "Yearly overview",
                    "Departments",
                    "Funding organisations",
                    "Recent-year pies",
                    "Tables",
                ]
            )

            with f1:
                st.plotly_chart(plot_overall_awards_by_year(project_filtered), use_container_width=True)

            with f2:
                if selected_departments:
                    include_funder = st.toggle(
                        "Add funding organisation to hover grouping",
                        value=True,
                        help="When on, bubbles are split by year, department, and funding organisation.",
                    )
                    st.plotly_chart(
                        plot_selected_departments_bubble(project_filtered, include_funder=include_funder),
                        use_container_width=True,
                    )
                department_options = sorted(project_filtered["Department"].dropna().unique())
                if department_options:
                    selected_single_department = st.selectbox(
                        "Department for annual funding bar chart",
                        department_options,
                        index=department_options.index("Geography - SoGE")
                        if "Geography - SoGE" in department_options
                        else 0,
                    )
                    st.plotly_chart(
                        plot_department_yearly_awards(project_filtered, selected_single_department),
                        use_container_width=True,
                    )

            with f3:
                color_by_department = st.toggle(
                    "Colour by department instead of funding organisation",
                    value=False,
                )
                st.plotly_chart(
                    plot_funding_org_bubble(project_filtered, color_by_department=color_by_department),
                    use_container_width=True,
                )

            with f4:
                st.caption("Donut charts are generated for up to six most recent years in the filtered data.")
                plot_recent_year_pies(project_filtered, max_years=6)

            with f5:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Top total funding by organisation and department")
                    st.dataframe(
                        top_funding_org_table(project_filtered).head(50),
                        use_container_width=True,
                        hide_index=True,
                    )
                with c2:
                    st.subheader("Highest average award by organisation and department")
                    st.dataframe(
                        average_awards_table(project_filtered).head(50),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.subheader("Filtered project records")
                st.dataframe(project_filtered, use_container_width=True, hide_index=True)

with success_tab:
    if success_df is None or success_df.empty:
        st.warning("No success-rate data is available.")
    else:
        success_filtered = apply_success_filters(
            success_df,
            selected_success_funders,
            success_year_range,
        )
        if success_filtered.empty:
            st.warning("No success-rate records match the current filters.")
        else:
            show_success_metrics(success_filtered)
            st.markdown(
                "<div class='section-note'>Status values are mapped as approved = A, rejected = U, and submitted = S. Success rate is Successful / Submitted.</div>",
                unsafe_allow_html=True,
            )

            s1, s2, s3, s4 = st.tabs(
                ["Success by funder", "Funder-year heatmap", "Scheme portfolio", "Tables"]
            )
            with s1:
                st.plotly_chart(plot_success_by_funder(success_filtered), use_container_width=True)
            with s2:
                st.plotly_chart(
                    plot_success_heatmap(success_filtered, top_n_funders=top_n_funders),
                    use_container_width=True,
                )
            with s3:
                scheme_fig = plot_scheme_portfolio(success_filtered)
                if scheme_fig is None:
                    st.info("No scheme-level records to plot.")
                else:
                    st.plotly_chart(scheme_fig, use_container_width=True)
            with s4:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Funder summary")
                    st.dataframe(get_funder_summary(success_filtered), use_container_width=True, hide_index=True)
                with c2:
                    st.subheader("Scheme summary")
                    st.dataframe(get_scheme_summary(success_filtered), use_container_width=True, hide_index=True)
                st.subheader("Filtered success records")
                st.dataframe(success_filtered, use_container_width=True, hide_index=True)

with about_tab:
    st.subheader("How this app is organised")
    st.markdown(
        """
        **Repository structure**

        ```text
        your-repo/
        ├── app.py
        ├── requirements.txt
        ├── README.md
        ├── .gitignore
        ├── .streamlit/
        │   └── config.toml
        └── data/
            ├── projectsearch.csv              
            └── success_rate_data.csv          
        ```

        """
    )
    st.subheader("Expected columns")
    st.markdown(
        """
        **Project funding CSV:** `LeadROName`, `FundingOrgName`, `Department`, `AwardPounds`, `ProjectId`, plus either `StartYear` or `StartDate`.

        **Success-rate CSV:** `Financial Year`, `Fund Deci Status Desc`, `Funder`, `Project Ref`, `Scheme Name`.
        """
    )
