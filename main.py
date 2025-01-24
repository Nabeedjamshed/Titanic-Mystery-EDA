# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import io

# df = pd.read_csv("titanic.csv")

# df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# df['HasCabin'] = df['Cabin'].notnull().astype(int)

# st.title("Titanic Mystery")

# st.sidebar.title("Choose Analysis")
# analysis = st.sidebar.selectbox("Select Analysis Type", [
#     "Overview",
#     "Gender Distribution",
#     "Survival Count",
#     "Passenger Class Analysis",
#     "Age Distribution",
#     "Fare Analysis",
#     "Family Size and Survival",
#     "Survival by Gender and Class",
#     "Embarkation Analysis",
#     "Custom Search/Filtering",
#     "Age vs Fare Scatter Plot",
#     "Correlation Heatmap",
#     "Interactive Fare Analysis (Plotly)"
# ])

# if analysis == "Overview":
#     st.subheader("Dataset Overview")
#     st.write("First 5 rows of the dataset:")
#     st.dataframe(df.head())

#     st.write("Basic info about the dataset:")
#     buffer = io.StringIO()
#     df.info(buf=buffer)
#     info = buffer.getvalue()
#     st.text(info)

#     st.write(f"Shape: {df.shape}")
#     st.write("Missing Values:")
#     st.write(df.isnull().sum())

# elif analysis == "Gender and Age Distribution":
#     st.subheader("Gender Distribution on Titanic")
#     count = df['Sex'].value_counts()
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.pie(count, labels=count.index, autopct='%1.1f%%', colors=['skyblue', 'pink'])
#     ax.set_title("Gender Distribution")
    
#     st.pyplot(fig)

#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(df['Age'], kde=True, ax=ax, color='purple', bins=20)
#     ax.set_title("Age Distribution")
#     st.pyplot(fig)

# elif analysis == "Survival Analysis":
#     st.subheader("Survival Count")
#     fig, ax = plt.subplots()
#     sns.countplot(x='Survived', data=df, ax=ax)
#     ax.set_title("Count of Survival (0 = Not Survived, 1 = Survived)")
#     st.pyplot(fig)

#     st.subheader("Family Size and Survival")
#     fig, ax = plt.subplots()
#     sns.barplot(x='FamilySize', y='Survived', data=df, ax=ax)
#     plt.title("Family Size and Survival")
#     st.pyplot(fig)

#     st.subheader("Survival by Gender and Passenger Class")
#     fig, ax = plt.subplots()
#     sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, ax=ax)
#     ax.set_title("Survival Rate by Gender and Passenger Class")
#     st.pyplot(fig)

# elif analysis == "Passenger Class and Fare Analysis":
#     st.subheader("Passenger Class Analysis")
#     fig, ax = plt.subplots()
#     sns.countplot(x='Pclass', data=df, ax=ax)
#     ax.set_title("Passenger Class Distribution")
#     st.pyplot(fig)

#     st.subheader("Fare Analysis")
#     fig, ax = plt.subplots()
#     sns.boxplot(x='Pclass', y='Fare', data=df, ax=ax)
#     ax.set_title("Fare Distribution by Passenger Class")
#     st.pyplot(fig)

# elif analysis == "Embarkation Analysis":
#     st.subheader("Embarkation Analysis")
#     embark_counts = df['Embarked'].value_counts()
#     st.write("Passengers by Embarkation Point (S: Southampton, C: Cherbourg, Q: Queenstown):")
#     st.bar_chart(embark_counts)
#     st.write("Survival Rate by Embarkation Point:")
#     fig, ax = plt.subplots()
#     sns.barplot(x='Embarked', y='Survived', data=df, ax=ax)
#     ax.set_title("Survival Rate by Embarkation Point")
#     st.pyplot(fig)

# elif analysis == "Custom Search/Filtering":
#     st.subheader("Custom Search/Filtering")
#     name_search = st.text_input("Search by Name (leave blank for all):")
#     gender_filter = st.selectbox("Filter by Gender", options=["All", "male", "female"])
#     pclass_filter = st.selectbox("Filter by Passenger Class", options=["All", 1, 2, 3])

#     filtered_df = df.copy()
#     if name_search:
#         filtered_df = filtered_df[filtered_df['Name'].str.contains(name_search, case=False, na=False)]  # We use .str.contains(...) in pandas because it is a powerful and flexible method to search for specific substrings in a string column of a DataFrame
#     if gender_filter != "All":
#         filtered_df = filtered_df[filtered_df['Sex'] == gender_filter]
#     if pclass_filter != "All":
#         filtered_df = filtered_df[filtered_df['Pclass'] == pclass_filter]

#     st.write(f"Filtered Results ({len(filtered_df)} passengers):")
#     st.dataframe(filtered_df)

# elif analysis == "Age vs Fare Scatter Plot":
#     st.subheader("Age vs Fare Scatter Plot")
#     fig, ax = plt.subplots()
#     sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, ax=ax, palette={0: "red", 1: "green"})
#     ax.set_title("Age vs Fare with Survival")
#     st.pyplot(fig)

# elif analysis == "Correlation Heatmap":
#     st.subheader("Correlation Heatmap")
#     num_df = df.select_dtypes([int, float])
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
#     plt.title("Correlation Heatmap")
#     st.pyplot(fig)

# elif analysis == "Interactive Fare Analysis":
#     st.subheader("Interactive Fare Analysis")
#     fig = px.bar(df, x='Pclass', y='Fare', color='Survived', title="Fare by Passenger Class and Survival")
#     st.plotly_chart(fig)


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io

# Load dataset
df = pd.read_csv("titanic.csv")

# Data preprocessing
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['HasCabin'] = df['Cabin'].notnull().astype(int)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Title
st.title("Titanic Mystery")

# Sidebar
st.sidebar.title("Choose Analysis")
st.sidebar.subheader("Exploratory Analysis")
analysis = st.sidebar.selectbox("Select Analysis Type", [
    "Overview",
    "Gender and Age Distribution",
    "Survival Analysis",
    "Passenger Class and Fare Analysis",
    "Embarkation Analysis",
    "Custom Search/Filtering",
    "Age vs Fare Scatter Plot",
    "Correlation Heatmap",
    "Interactive Fare Analysis"
])

# Analysis options
if analysis == "Overview":
    st.subheader("Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    st.write("Basic info about the dataset:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    st.text(info)

    st.write(f"Shape: {df.shape}")
    st.write("Missing Values:")
    st.write(df.isnull().sum())

elif analysis == "Gender and Age Distribution":
    st.subheader("Gender Distribution")
    count = df['Sex'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(count, labels=count.index, autopct='%1.1f%%', colors=['skyblue', 'pink'])
    ax.set_title("Gender Distribution")
    st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde=True, ax=ax, color='purple', bins=20)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

elif analysis == "Survival Analysis":
    st.subheader("Survival Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=df, ax=ax)
    ax.set_title("Count of Survival (0 = Not Survived, 1 = Survived)")
    st.pyplot(fig)

    st.subheader("Family Size and Survival")
    fig, ax = plt.subplots()
    sns.barplot(x='FamilySize', y='Survived', data=df, ax=ax)
    ax.set_title("Family Size and Survival")
    st.pyplot(fig)

    st.subheader("Survival by Gender and Passenger Class")
    fig, ax = plt.subplots()
    sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, ax=ax)
    ax.set_title("Survival Rate by Gender and Passenger Class")
    st.pyplot(fig)

elif analysis == "Passenger Class and Fare Analysis":
    st.subheader("Passenger Class Analysis")
    fig, ax = plt.subplots()
    sns.countplot(x='Pclass', data=df, ax=ax)
    ax.set_title("Passenger Class Distribution")
    st.pyplot(fig)

    st.subheader("Fare Analysis")
    fig, ax = plt.subplots()
    sns.boxplot(x='Pclass', y='Fare', data=df, ax=ax)
    ax.set_title("Fare Distribution by Passenger Class")
    st.pyplot(fig)

elif analysis == "Embarkation Analysis":
    st.subheader("Embarkation Analysis")
    embark_counts = df['Embarked'].value_counts()
    st.write("Passengers by Embarkation Point (S: Southampton, C: Cherbourg, Q: Queenstown):")
    st.bar_chart(embark_counts)

    st.write("Survival Rate by Embarkation Point:")
    fig, ax = plt.subplots()
    sns.barplot(x='Embarked', y='Survived', data=df, ax=ax)
    ax.set_title("Survival Rate by Embarkation Point")
    st.pyplot(fig)

elif analysis == "Custom Search/Filtering":
    st.subheader("Custom Search/Filtering")
    name_search = st.text_input("Search by Name (leave blank for all):")
    gender_filter = st.selectbox("Filter by Gender", options=["All", "male", "female"])
    pclass_filter = st.selectbox("Filter by Passenger Class", options=["All", 1, 2, 3])
    min_age, max_age = st.slider("Select Age Range", 0, int(df['Age'].max()), (0, 80))

    filtered_df = df.copy()
    if name_search:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(name_search, case=False, na=False)]
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df['Sex'] == gender_filter]
    if pclass_filter != "All":
        filtered_df = filtered_df[filtered_df['Pclass'] == pclass_filter]
    filtered_df = filtered_df[(filtered_df['Age'] >= min_age) & (filtered_df['Age'] <= max_age)]

    st.write(f"Filtered Results ({len(filtered_df)} passengers):")
    st.dataframe(filtered_df)

    # Download filtered data
    st.download_button(
        "Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name='filtered_data.csv',
        mime='text/csv'
    )

elif analysis == "Age vs Fare Scatter Plot":
    st.subheader("Age vs Fare Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, ax=ax, palette={0: "red", 1: "green"})
    ax.set_title("Age vs Fare with Survival")
    st.pyplot(fig)

elif analysis == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes([int, float])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

elif analysis == "Interactive Fare Analysis":
    st.subheader("Interactive Fare Analysis")
    min_fare, max_fare = st.slider("Select Fare Range", 0, int(df['Fare'].max()), (0, 500))
    filtered_df = df[(df['Fare'] >= min_fare) & (df['Fare'] <= max_fare)]
    fig = px.bar(filtered_df, x='Pclass', y='Fare', color='Survived', title="Filtered Fare Analysis")
    st.plotly_chart(fig)
