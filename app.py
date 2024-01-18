import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st

def add_price_range_and_visualize_contribution(df, col_name, num_bins=4):
    X = df[[col_name]].copy()
    df.reset_index(inplace=True)
    num_clusters = num_bins

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    X['Cluster'] = kmeans.fit_predict(X)

    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

    cluster_centers = np.round(cluster_centers).astype(int)

    bins = np.concatenate(([-np.inf], cluster_centers, [np.inf]))

    labels = [f"Under ${cluster_centers[0]}" if i == 0
              else f"Over ${cluster_centers[-1]}" if i == len(bins) - 2
              else f"${int(bins[i])}-${int(bins[i+1])}" for i in range(len(bins)-1)]

    df['Price Range'] = pd.cut(df[col_name], bins=bins, labels=labels, right=False)
    
    
    st.subheader("Number of Units sold in each Price Range")
    df_sorted = df.sort_values(by='Price Range')
    hist_fig, hist_ax = plt.subplots()
    hist_ax.hist(df_sorted['Price Range'], bins=len(labels))  # Use the number of labels as the number of bins

    # Customize x-axis ticks
    hist_ax.set_xticks(range(len(labels)))
    hist_ax.set_xticklabels(labels, rotation=45, ha='right')

    st.pyplot(hist_fig)

    total_sales_by_range = df.groupby('Price Range')[col_name].sum()

    contribution_percentage = (total_sales_by_range / df[col_name].sum()) * 100

    # Visualize the contribution
    st.subheader("Contribution of Each Price Range to Total Sales")
    fig, ax = plt.subplots()
    ax.bar(contribution_percentage.index, contribution_percentage)
    ax.set_xlabel('Price Range')
    ax.set_ylabel('Contribution (%)')
    ax.set_title('Contribution of Each Price Range to Total Sales')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    st.pyplot(fig)
    
    return df



def main():
    st.title("Price Range Prediction")

    num_bins = st.slider("Select the number of bins:", min_value=2, max_value=20, value=4)

    col_name = st.text_input("Enter the column name:", "Retail Price")

    sheet_name = st.text_input("Enter the sheet name:", "Sheet1")

    header = st.number_input("Enter headers to skip if any:",0)

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header)

        st.subheader("Original Data")
        st.write(df)

        df_result = add_price_range_and_visualize_contribution(df, col_name, num_bins)

        st.subheader("Data with Price Range")
        st.write(df_result)

        st.markdown("### Download Processed Data")
        st.write("Click below to download the processed data.")
        st.download_button(
            label="Download Processed Data",
            data=df_result.to_csv(index=False).encode(),
            file_name=f"processed_data.csv",
            key="download_button",
        )


if __name__ == "__main__":
    main()
