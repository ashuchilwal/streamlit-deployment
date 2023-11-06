import streamlit as st
import pandas as pd
import string
import re
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from statsmodels.tsa.stattools import adfuller

# import nltk
# nltk.download('punkt')
# from nltk.corpus import stopwords

df = None



def handle_null_values(input_df):
    df = input_df.dropna()
    return df


def convert_to_lowercase(input_df):
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].str.lower()
    return input_df

def remove_stopwords(input_df):
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].apply(stopeword_remove)
    return input_df


def stopeword_remove(text):
    stop_words = set([
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
                "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
                "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
                "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y"
            ])
    words = text.split()
    cleaned_text = " ".join([word for word in words if word.lower() not in stop_words])
    return cleaned_text

def dependent_independent_feature_selection(input_df):
    st.write('Please Select Dependent and Independent Columns')

    all_columns = input_df.columns.tolist()
    select_dependent_column = st.selectbox(
        'Select Dependent Column', all_columns)
    # input_df = input_df.drop(select_dependent_column, axis=1)

    all_columns = input_df.columns.tolist()
    selected_columns_d = st.multiselect('Select', all_columns)
    if selected_columns_d is not None:
        new_df = input_df[selected_columns_d]
        return (new_df, select_dependent_column)
    else:
        st.warning('Please select atleast one column')

    return None, None

# Function to remove punctuation from text


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = ' '.join(text.split())
    return text


def datatype_converter(df, conversions):
    # converted_df = df.copy()
    for col, new_type in conversions.items():
        try:
            if new_type == 'datetime64':
                df[col] = pd.to_datetime(df[col])
            elif new_type == 'object' and df[col].dtype == 'datetime64':
                # df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                df[col] = df[col].dt.strftime('%Y-%m-%d')
            else:
                # Attempt to convert the column to the selected data type
                df[col] = df[col].astype(new_type)
        except ValueError as e:
            st.warning(f"Conversion error in column '{col}': {str(e)}")

    return df


def numeric_values_extract(text):
    # Use regular expressions to find numeric values
    numeric_values = re.findall(r'\d+', text)
    if numeric_values:
        return int(numeric_values[0])
    else:
        return None

# ******************************************


def remove_space_punctuation(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(remove_punctuation)
    return df


def extract_numeric_values(df):
    st.subheader("Select columns containing numeric values:")
    selected_column = st.multiselect("Choose columns:", df.columns)
    if selected_column:
        for col in selected_column:
            df[col] = df[col].apply(numeric_values_extract)
    return df


def change_datatype(df):
    st.subheader("Feature Information (Pre Datatype)")
    st.write(df.dtypes)

    all_data_types = ['int64', 'float',  'bool', 'object', 'datetime64']
    # Create a dictionary to map column names to selected data types
    data_type_conversions = {}
    # Display dropdowns for data type selection
    st.header("Select Data Types")
    for col in df.columns:
        selected_data_type = st.selectbox(
            f"Select data type for {col}",
            options=all_data_types,
            index=all_data_types.index(df[col].dtype),
            format_func=lambda x: x if isinstance(
                x, str) else x.__name__,  # Display data type names
        )
        data_type_conversions[col] = selected_data_type
    # Convert data types based on user selection
    df = datatype_converter(df, data_type_conversions)
    st.subheader("After datatype conversion Feature Information")
    return df


def feature_scaling(input_df):
    numeric_feature = []
    for col in input_df.columns:
        if input_df[col].dtype == 'int64' or input_df[col].dtype == 'float64':
            numeric_feature.append(col)

    st.sidebar.subheader("Scale Numeric Variables")
    scaler = StandardScaler()
    scaled_df = input_df.copy()
    for feature in numeric_feature:
        scaled_df[feature +
                  "_scaled"] = scaler.fit_transform(input_df[[feature]])

    scaled_df.drop(numeric_feature, axis=1, inplace=True)
    return scaled_df


def one_hot_encoding(input_df):
    categorical_feature = []
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            categorical_feature.append(col)

    st.subheader('Select Columns for one hot encoding')
    selected_columns = st.multiselect("Choose Columns", categorical_feature)

    st.sidebar.subheader("Encode Categorical Variables")
    df_encoded = pd.get_dummies(
        input_df, columns=selected_columns, drop_first=True)
    return df_encoded


def label_encoding(input_df):
    st.subheader("Select columns for label encoding:")
    selected_columns = st.multiselect("Choose columns:", input_df.columns)
    label_encoder = LabelEncoder()
    # input_df[dependent_column] = label_encoder.fit_transform(input_df[dependent_column])
    # return input_df[dependent_column]
    for col in selected_columns:
        input_df[col] = label_encoder.fit_transform(input_df[col])
    return input_df

# time series


def convert_dataset(df):
    all_columns = df.columns.tolist()
    datetime_column = st.selectbox(
        'Select Datetime Column', all_columns)

    # st.write('Please Select the other columns')
    # other_columns = st.multiselect('Select Other Columns', all_columns)

    conversion_options = ['Daily', 'Weekly', 'Monthly', 'Yearly']
    selected_option = st.selectbox(
        'Select the desired dataset:', conversion_options)

    # df[datetime_column] = pd.to_datetime(df[datetime_column])
    df = df.set_index(datetime_column)
    # df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')

    if selected_option == 'Daily':
        daily_df = df[df.columns].resample('D').sum()
        return daily_df
    elif selected_option == 'Weekly':
        df_weekly = df[df.columns].resample('W').sum()
        return df_weekly
    elif selected_option == 'Monthly':
        df_monthly = df[df.columns].resample('M').sum()
        return df_monthly
    elif selected_option == 'Yearly':
        df_yearly = df[df.columns].resample('Y').sum()
        return df_yearly


def Check_stationarity(input_df):
    dftest = adfuller(input_df, autolag='AIC')
    st.write("1. ADF : ", dftest[0])
    st.write("2. P-Value : ", dftest[1])
    st.write("3. Num of Lags : ", dftest[2])
    st.write(
        "4. Num of Observations Used For ADF Regeression and Critical Values Calculation : ", dftest[3])
    st.write("5. Critical Values : ")
    for key, val in dftest[4].items():
        st.write("\t", key, " : ", val)
    if dftest[1] <= 0.05:
        return input_df
    else:
        st.write('After Perform diff')
        input_df = input_df.diff().dropna()
        dftest = adfuller(input_df, autolag='AIC')
        st.write("1. ADF : ", dftest[0])
        st.write("2. P-Value : ", dftest[1])
        st.write("3. Num of Lags : ", dftest[2])
        st.write(
            "4. Num of Observations Used For ADF Regeression and Critical Values Calculation : ", dftest[3])
        st.write("5. Critical Values : ")
        for key, val in dftest[4].items():
            st.write("\t", key, " : ", val)
        return input_df


def download_processed_df(df_to_download):
    st.subheader("Download Processed Dataset")
    st.download_button("Download CSV", df_to_download.to_csv().encode("utf-8"),
                       file_name="processed_dataset.csv",
                       key="download_processed_csv")

# ******************************************


def main():
    st.title("Feature Engineering App")

    # Upload dataset
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        global df
        # Read the uploaded dataset
        st.sidebar.subheader("Upload Dataset")
        df = pd.read_csv(uploaded_file)
        df_copy = df.copy()

        st.sidebar.success("File uploaded successfully!")
        st.sidebar.write('Original Data')
        st.sidebar.write(df_copy.head())

        df = handle_null_values(df)

        if st.checkbox("Data Cleaning"):
            if st.checkbox("Convert text into Lowercase"):
                df = convert_to_lowercase(df)

            if st.checkbox('Remove Extra Spaces and Punctuation'):
                df = remove_space_punctuation(df)

            if st.checkbox('Extract Numeric Values'):
                df = extract_numeric_values(df)

            if st.checkbox('Remove Stopwords'):
                df = remove_stopwords(df)

        if st.checkbox("Regression OR Classification"):

            st.subheader("Feature Selection")
            df, dependent_column = dependent_independent_feature_selection(df)

            if st.checkbox('Change Datatype'):
                df = change_datatype(df)
                st.write(df.dtypes)

            if st.checkbox('Normalization'):
                df = feature_scaling(df)

            if st.checkbox('Label Encoding'):
                df = label_encoding(df)

            if st.checkbox('One Hot Encoding'):
                df = one_hot_encoding(df)

        elif st.checkbox("Time Series"):
            if st.checkbox('Change Datatype'):
                df = change_datatype(df)
                st.write(df.dtypes)

            if st.checkbox('Convert Dataset into Weekly, Monthly, Yearly'):
                df = convert_dataset(df)

            if st.checkbox('Check Stationarity'):
                df = Check_stationarity(df)

        st.sidebar.write('Dataset View')
        st.sidebar.write(df)

        download_processed_df(df)


if __name__ == '__main__':
    main()
