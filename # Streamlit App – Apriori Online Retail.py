# Streamlit App – Apriori Online Retail

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Apriori Online Retail", layout="wide")

st.title("🛒 Market Basket Analysis dengan Apriori")
st.write("Analisis association rules pada dataset Online Retail menggunakan algoritma Apriori.")

# Upload file
uploaded_file = st.file_uploader(
    "Upload file Online_Retail.xlsx",
    type=["xlsx"]
)

if uploaded_file is not None:

    # =========================
    # Load Dataset
    # =========================
    df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Awal")
    st.write(df.head())
    st.write(f"Shape data: {df.shape}")

    # =========================
    # Data Cleaning
    # =========================
    df_clean = df.dropna(subset=['CustomerID', 'Description'])
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean['Description'] = df_clean['Description'].str.strip()

    st.subheader("Data Setelah Cleaning")
    st.write(df_clean.head())
    st.write(f"Shape setelah cleaning: {df_clean.shape}")

    # =========================
    # Top Produk
    # =========================
    st.subheader("Top 10 Produk")

    top_products = (
        df_clean.groupby('Description')['Quantity']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    top_products.sort_values().plot(kind='barh', ax=ax)
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Produk")
    st.pyplot(fig)

    # =========================
    # Filter Country
    # =========================
    country = st.selectbox(
        "Pilih Negara",
        sorted(df_clean['Country'].unique())
    )

    df_country = df_clean[df_clean['Country'] == country]

    st.write(f"Jumlah transaksi di {country}: {df_country['InvoiceNo'].nunique()}")

    # =========================
    # Basket Transaction
    # =========================
    basket = (
        df_country.groupby('InvoiceNo')['Description']
        .apply(list)
        .tolist()
    )

    # =========================
    # Transaction Encoding
    # =========================
    te = TransactionEncoder()
    te_array = te.fit(basket).transform(basket)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    st.subheader("Encoded Transaction Matrix")
    st.write(df_encoded.head())
    st.write(f"Dimensi matrix: {df_encoded.shape}")

    # =========================
    # Sidebar Parameter
    # =========================
    st.sidebar.header("Parameter Apriori")

    min_support = st.sidebar.slider(
        "Min Support",
        min_value=0.001,
        max_value=0.1,
        value=0.015,
        step=0.001
    )

    min_confidence = st.sidebar.slider(
        "Min Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05
    )

    max_len = st.sidebar.slider(
        "Max Itemset Length",
        min_value=2,
        max_value=5,
        value=3
    )

    # =========================
    # Apriori
    # =========================
    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )

    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

    st.subheader("Frequent Itemsets")
    st.write(frequent_itemsets.head(20))
    st.write(f"Total itemsets ditemukan: {len(frequent_itemsets)}")

    # =========================
    # Association Rules
    # =========================
    rules = association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=min_confidence
    )

    if len(rules) > 0:

        rules = rules.sort_values('lift', ascending=False)

        st.subheader("Association Rules")

        display_rules = rules[[
            'antecedents',
            'consequents',
            'support',
            'confidence',
            'lift'
        ]].copy()

        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))

        st.dataframe(display_rules.head(20))

        # =========================
        # Scatter Plot
        # =========================
        st.subheader("Visualisasi Rules")

        fig2, ax2 = plt.subplots(figsize=(8, 6))

        scatter = ax2.scatter(
            rules['support'],
            rules['confidence'],
            c=rules['lift'],
            alpha=0.7
        )

        ax2.set_xlabel('Support')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Support vs Confidence')

        plt.colorbar(scatter, ax=ax2, label='Lift')

        st.pyplot(fig2)

        # =========================
        # Recommendation System
        # =========================
        st.subheader("Rekomendasi Produk")

        product_input = st.text_input(
            "Masukkan nama produk",
            value="ROSES"
        )

        if product_input:

            recommendations = []

            for _, row in rules.iterrows():
                antecedents = [item.lower() for item in row['antecedents']]

                if any(product_input.lower() in item for item in antecedents):
                    recommendations.append({
                        'Jika membeli': ', '.join(list(row['antecedents'])),
                        'Maka membeli': ', '.join(list(row['consequents'])),
                        'Confidence': round(row['confidence'], 3),
                        'Lift': round(row['lift'], 3)
                    })

            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                st.dataframe(rec_df.head(10))
            else:
                st.warning("Tidak ada rekomendasi ditemukan.")

    else:
        st.warning("Tidak ada association rules ditemukan. Coba turunkan parameter support/confidence.")

else:
    st.info("Silakan upload dataset Online_Retail.xlsx terlebih dahulu.")
```

## Cara Menjalankan

1. Install library:

```bash
pip install streamlit pandas matplotlib mlxtend openpyxl
```

2. Simpan file dengan nama:

```bash
app.py
```

3. Jalankan Streamlit:

```bash
streamlit run app.py
```
