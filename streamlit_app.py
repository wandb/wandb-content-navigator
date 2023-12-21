import streamlit as st
import pandas as pd
import ast

# Load the data
df = pd.read_csv('merged_df.csv')

# Convert string representation of list to actual list
list_columns = ['value_props', 'application', 'integrations', 'audience']  # add here all columns that are lists
for col in list_columns:
    df[col] = df[col].apply(ast.literal_eval)

# Create a multiselect widget for each column that you want to filter by
use_case_options = df['use_case'].unique()
value_props_options = list(set([item for sublist in df['value_props'].tolist() for item in sublist]))
application_options = list(set([item for sublist in df['application'].tolist() for item in sublist]))
integrations_options = list(set([item for sublist in df['integrations'].tolist() for item in sublist]))
audience_options = list(set([item for sublist in df['audience'].tolist() for item in sublist]))

selected_use_case = st.sidebar.multiselect('Use Case', options=use_case_options)
selected_value_props = st.sidebar.multiselect('Value Propositions', options=value_props_options)
selected_application = st.sidebar.multiselect('Application', options=application_options)
selected_integrations = st.sidebar.multiselect('Integrations', options=integrations_options)
selected_audience = st.sidebar.multiselect('Audience', options=audience_options)

# Filter the dataframe based on the selected values
if selected_use_case or selected_value_props or selected_application or selected_integrations or selected_audience:
    filtered_df = df[(df['use_case'].isin(selected_use_case) if selected_use_case else True) &
                     (df['value_props'].apply(lambda x: any([k in x for k in selected_value_props])) if selected_value_props else True) &
                     (df['application'].apply(lambda x: any([k in x for k in selected_application])) if selected_application else True) &
                     (df['integrations'].apply(lambda x: any([k in x for k in selected_integrations])) if selected_integrations else True) &
                     (df['audience'].apply(lambda x: any([k in x for k in selected_audience])) if selected_audience else True)]

    # Display the filtered dataframe
    for _, row in filtered_df.iterrows():
        st.markdown(f"**Title:** {row['display_name']}")
        st.markdown(f"**Summary:** {row['summary']}")
        st.markdown(f"**URL:** [link]({row['source']})")
        st.markdown(f"**Use Case:** {row['use_case']}")
        st.markdown(f"**Value Props:** {row['value_props']}")
        st.markdown(f"**Application:** {row['application']}")
        st.markdown(f"**Integrations:** {row['integrations']}")
        st.markdown(f"**Audience:** {row['audience']}")
        st.markdown('---')
        st.write(row)

else:
    # if no filters are selected, display a message to the user to select filters
    st.write('Please select filters to start exploring the data')
