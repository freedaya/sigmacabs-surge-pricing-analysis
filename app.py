import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sigma Cabs | EDA Dashboard",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# LIGHT THEME STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .card-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #4361ee;
        margin-bottom: 6px;
    }
    .card-title {
        font-size: 15px;
        font-weight: 700;
        color: #212529;
        margin-bottom: 8px;
    }
    .card-body {
        font-size: 14px;
        color: #495057;
        line-height: 1.7;
    }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #212529;
        border-left: 4px solid #4361ee;
        padding-left: 12px;
        margin: 28px 0 14px 0;
    }
    .q-badge {
        display: inline-block;
        background: #e8ecfd;
        color: #4361ee;
        border: 1px solid #b6c1f8;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .analysis-box {
        background: #f0f4ff;
        border-left: 3px solid #4361ee;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-top: 14px;
        font-size: 14px;
        color: #495057;
        line-height: 1.8;
    }
    .analysis-box b { color: #212529; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 800;
        color: #4361ee;
    }
    .metric-lbl {
        font-size: 11px;
        color: #6c757d;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .coming-soon {
        text-align: center;
        padding: 80px 20px;
        color: #adb5bd;
        font-size: 16px;
    }
    .coming-soon span {
        font-size: 48px;
        display: block;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("sigma_cabs.csv")
    df.drop("Trip_ID", axis=1, inplace=True)
    df_clean = df.copy()
    df_clean.drop('Var1', axis=1, inplace=True)
    df_clean["Type_of_Cab"].fillna(df_clean["Type_of_Cab"].mode()[0], inplace=True)
    df_clean["Confidence_Life_Style_Index"].fillna(
        df_clean["Confidence_Life_Style_Index"].mode()[0], inplace=True)
    df_clean["Customer_Since_Months"].fillna(
        df_clean["Customer_Since_Months"].median(), inplace=True)
    df_clean["Life_Style_Index"].fillna(
        df_clean["Life_Style_Index"].median(), inplace=True)
    df_clean['Surge_Pricing_Type'] = df_clean['Surge_Pricing_Type'].astype(str)
    return df, df_clean

df, df_clean = load_data()


@st.cache_data
def get_corr_matrix(_df_clean):
    df_temp = _df_clean.copy()
    df_temp['Type_of_Cab'] = df_temp['Type_of_Cab'].map(
        {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4})
    df_temp['Destination_Type'] = df_temp['Destination_Type'].map(
        {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,
         'I':8,'J':9,'K':10,'L':11,'M':12,'N':13})
    df_temp['Confidence_Life_Style_Index'] = df_temp['Confidence_Life_Style_Index'].map(
        {'A': 0, 'B': 1, 'C': 2})
    df_temp = pd.get_dummies(df_temp, columns=['Gender'], prefix='Gender')
    df_temp['Surge_Pricing_Type'] = df_temp['Surge_Pricing_Type'].astype(float)
    cols = ['Surge_Pricing_Type'] + [c for c in df_temp.columns if c != 'Surge_Pricing_Type']
    return df_temp[cols].corr()

corr_matrix = get_corr_matrix(df_clean)

COLORS = {'1': '#636EFA', '2': '#EF553B', '3': '#00CC96'}
SURGE_ORDER = {"Surge_Pricing_Type": ["1", "2", "3"]}
PLOTLY_LAYOUT = dict(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(
        family="Arial, sans-serif",
        size=16, 
        color='#212529'
    ),
    title=dict(
        font=dict(size=24, color='#212529'),
        x=0,
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.8)',
        font=dict(size=20),
        bordercolor='#dee2e6',
        borderwidth=1
    ),
    xaxis=dict(
        showgrid=False, 
        linecolor='#dee2e6',
        tickfont=dict(size=20),
        title=dict(
            font=dict(size=24, color='#212529')
        )
    ),
    yaxis=dict(
        gridcolor='#f1f3f5', 
        linecolor='#dee2e6',
        tickfont=dict(size=20),
        title=dict(
            font=dict(size=24, color='#212529')
        )
    ),
    margin=dict(l=80, r=50, t=100, b=80) 
)


# ─────────────────────────────────────────────
# CHART FUNCTIONS — PLOTLY
# ─────────────────────────────────────────────

def def_num(df, col, title=None):
    if title is None:
        title = f"Distribusi {col} berdasarkan Surge Pricing"
    fig = px.violin(
        df, x="Surge_Pricing_Type", y=col,
        color="Surge_Pricing_Type",
        points="outliers", box=True,
        category_orders=SURGE_ORDER,
        color_discrete_map=COLORS,
        title=title, height=1000
    )
    fig.update_layout(
        violingap=0,
        xaxis_title="Surge Pricing Type",
        yaxis_title=col,
        showlegend=False,
        **PLOTLY_LAYOUT
    )
    return fig


def def_cat(df, col, sort_by_surge3=False, title=None):
    ct_count = pd.crosstab(df[col], df["Surge_Pricing_Type"])
    sorted_cat = ct_count.index.tolist()
    if sort_by_surge3:
        ct_pct = ct_count.div(ct_count.sum(axis=1), axis=0)
        sorted_cat = ct_pct.sort_values(by='3', ascending=False).index.tolist()
    ct_percentage = ct_count.div(ct_count.sum(axis=1), axis=0) * 100
    ct_plot = ct_percentage.reset_index().melt(
        id_vars=col, var_name="Surge_Pricing_Type", value_name="percentage")
    ct_count_melt = ct_count.reset_index().melt(
        id_vars=col, var_name="Surge_Pricing_Type", value_name="count")
    ct_plot = ct_plot.merge(ct_count_melt, on=[col, "Surge_Pricing_Type"])
    if title is None:
        title = f"Distribusi Proporsi Surge Pricing berdasarkan {col} (%)"
    fig = px.bar(
        ct_plot, x=col, y="percentage",
        color="Surge_Pricing_Type",
        barmode="relative", text="percentage",
        category_orders={col: sorted_cat, "Surge_Pricing_Type": ["1", "2", "3"]},
        color_discrete_map=COLORS,
        title=title, height=1000
    )
    for trace in fig.data:
        surge_type = trace.name
        filtered = ct_plot[ct_plot["Surge_Pricing_Type"] == surge_type].copy()
        filtered = filtered.set_index(col).reindex(trace.x)
        trace.customdata = filtered[["count"]].values
        trace.hovertemplate = (
            f"{col}: <b>%{{x}}</b><br>"
            "Surge Type: %{fullData.name}<br>"
            "Persentase: %{y:.1f}%<br>"
            "Jumlah: %{customdata[0]}<extra></extra>"
        )
    fig.update_traces(
    texttemplate='%{text:.1f}%',
    textposition='inside',      
    insidetextanchor='middle',
    textfont=dict(size=18, color="white")
    )
    fig.update_layout(
        yaxis_title="Persentase (%)", xaxis_title="",
        uniformtext_mode='hide', uniformtext_minsize=8,
        **PLOTLY_LAYOUT
    )
    return fig


def get_univariate_plot(df, col):
    if df[col].dtype in ['int64', 'float64']:
        data = df[col].dropna()
        fig = px.histogram(
            df, x=col, nbins=50,
            title=f"Distribusi {col}",
            marginal="violin", height=1000,
            color_discrete_sequence=['#636EFA']
        )
        mu, std = data.mean(), data.std()
        x_range = np.linspace(data.min(), data.max(), 200)
        y_norm = norm.pdf(x_range, mu, std)
        bin_width = (data.max() - data.min()) / 50
        y_norm_scaled = y_norm * len(data) * bin_width
        
        fig.add_trace(go.Scatter(
            x=x_range, y=y_norm_scaled,
            mode='lines', name='Normal Fit',
            line=dict(color='#00CC96', width=4)
        ))
    else:
        df_counts = df[col].value_counts().reset_index()
        df_counts.columns = [col, 'count']
        
        fig = px.bar(
            df_counts, x=col, y='count',
            text='count',
            title=f"Distribusi {col}",
            color=col,
            height=1000
        )
        
        fig.update_traces(
            textposition='inside',   
            textfont_color='white',   
            textfont_size=18,         
            insidetextanchor='middle', 
            texttemplate='%{text:,}' 
        )

    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
    return fig

def heatmap_corr_plotly(corr_matrix):
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title="Heatmap Korelasi Antar Fitur & Target (Setelah Encoding)",
        aspect="auto",
        height=1200
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        xaxis_tickangle=-90,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(t=80, b=120),
        paper_bgcolor='white',
        font_color='#212529',
        title_font_size=20, 
        font_size=14,
    )
    
    fig.update_xaxes(
        tickfont_size=18,   
        title_font_size=20    
    )
    fig.update_yaxes(
        tickfont_size=18,
        title_font_size=20 
    )
    
    fig.update_traces(
        textfont_size=18,            
        textfont_family='Arial' 
    )
    
    return fig


def corr_bar_plotly(corr_matrix):
    surge_corr = corr_matrix['Surge_Pricing_Type'].drop('Surge_Pricing_Type')
    
    df_corr = pd.DataFrame({
        'Fitur': surge_corr.index,
        'Korelasi': surge_corr.values,
        'Abs_Korelasi': surge_corr.abs().values
    })
    
    df_corr = df_corr.sort_values('Abs_Korelasi', ascending=False)
    
    df_corr['Warna'] = df_corr['Korelasi'].apply(lambda x: 'Negatif' if x < 0 else 'Positif')
    
    fig = px.bar(
        df_corr,
        x='Korelasi',
        y='Fitur',
        orientation='h',
        color='Warna',
        color_discrete_map={'Negatif': '#EF553B', 'Positif': '#636EFA'},
        text='Korelasi',
        title="Korelasi Fitur vs Surge_Pricing_Type",
        height=1000,
        category_orders={'Fitur': df_corr['Fitur'].tolist()} 
    )
    
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside',
        textfont_size=18
    )
    
    fig.update_layout(
        showlegend=False,
        **PLOTLY_LAYOUT
    )
    
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 4px 0 20px 0;'>
        <div style='font-size: 20px; font-weight: 800; color: #212529;'>🚖 Sigma Cabs</div>
        <div style='font-size: 12px; color: #6c757d; margin-top: 4px;'>EDA Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    role = st.radio("Track", ["Data Analyst", "Data Science"], index=0)


# ═════════════════════════════════════════════
# DATA ANALYST — SINGLE SCROLL PAGE
# ═════════════════════════════════════════════
if role == "Data Analyst":

    st.markdown("<h1 style='color:#212529; margin-bottom:4px;'>Sigma Cabs — Surge Pricing Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d; font-size:15px; margin-bottom:24px;'>Data Analyst Track · Exploratory Data Analysis</p>", unsafe_allow_html=True)

    # ════════════════════
    # SECTION 1: OVERVIEW
    # ════════════════════
    st.markdown("<div class='section-header'>📋 Overview</div>", unsafe_allow_html=True)

    total_trips = f"{len(df_clean):,}"
    avg_dist = f"{df_clean['Trip_Distance'].mean():.1f} km"
    high_surge_pct = f"{(df_clean['Surge_Pricing_Type'] == '3').mean() * 100:.1f}%"
    avg_rating = f"{df_clean['Customer_Rating'].mean():.2f} / 5"

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (total_trips, "Total Trips", "🚗"),
        (avg_dist, "Avg. Distance", "📏"),
        (high_surge_pct, "High Surge (Type 3)", "⚡"),
        (avg_rating, "Avg. Rating", "⭐")
    ]

    for col_obj, (val, lbl, icon) in zip([c1, c2, c3, c4], metrics):
        with col_obj:
            st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 24px; margin-bottom: 5px;'>{icon}</div>
                    <div class='metric-val'>{val}</div>
                    <div class='metric-lbl'>{lbl}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""<div class='card'>
            <div class='card-label'>Project Background</div>
            <div class='card-body'>Sigma Cabs merupakan platform agregator layanan transportasi di India yang menghubungkan pengguna dengan berbagai penyedia jasa taksi. Tantangan terbesar yang dihadapi adalah fluktuasi harga atau <b>surge pricing</b> yang ditentukan secara dinamis. Selama hampir satu tahun beroperasi, perusahaan masih kesulitan memetakan faktor-faktor dominan yang memicu kenaikan skor surge pricing (skala 1–3).</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class='card'>
            <div class='card-label'>Business Problem</div>
            <div class='card-body'>Ketidakmampuan dalam memahami faktor yang mempengaruhi surge pricing dapat menyebabkan <b>ketidakefisienan</b> dalam mencocokkan pelanggan dengan layanan taksi yang sesuai. Hal ini berpotensi menurunkan kepuasan pelanggan serta efisiensi operasional perusahaan.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='card' style='border-left: 3px solid #4361ee;'>
        <div class='card-label'>Problem Statement</div>
        <div class='card-body' style='font-size:15px; color:#212529; font-style:italic;'>"Apa saja faktor utama yang mempengaruhi Surge Pricing Type, dan bagaimana pola perjalanan serta karakteristik pelanggan dapat digunakan untuk memahami dan mengantisipasi kenaikan harga?"</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
        <div class='card-label'>Tujuan Proyek</div>
        <div class='card-body'>Menghasilkan insight berbasis data untuk mengidentifikasi faktor-faktor utama yang mempengaruhi surge pricing, sehingga dapat membantu perusahaan dalam meningkatkan <b>strategi pricing</b>, optimasi supply-demand, serta meningkatkan pengalaman pelanggan.</div>
    </div>""", unsafe_allow_html=True)

    questions = [
        ("Q1", "Apakah jarak perjalanan mempengaruhi surge pricing?"),
        ("Q2", "Apakah tipe cab berpengaruh terhadap surge pricing?"),
        ("Q3", "Apakah customer baru dan lama memiliki pola surge yang berbeda?"),
        ("Q4", "Bagaimana lifestyle customer mempengaruhi harga?"),
        ("Q5", "Apakah jenis destinasi mempengaruhi surge pricing?"),
        ("Q6", "Bagaimana pengaruh rating customer terhadap harga?"),
        ("Q7", "Bagaimana perilaku pembatalan mempengaruhi kenaikan harga?"),
        ("Q8", "Apakah proporsi gender terhadap surge pricing setara?"),
    ]
    q_cols = st.columns(4)
    for i, (qn, qt) in enumerate(questions):
        with q_cols[i % 4]:
            st.markdown(f"""<div class='card' style='padding:12px 16px;'>
                <div class='q-badge'>{qn}</div>
                <div style='font-size:13px; color:#495057; line-height:1.5; margin-top:4px;'>{qt}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("📄 Dataset Preview & Data Dictionary"):
        n_rows = st.slider("Jumlah baris yang ditampilkan", min_value=5, max_value=1000, value=10, step=5)
        st.markdown(f"**Dataset Preview (Random Sample)**")
        st.dataframe(df.sample(n_rows), use_container_width=True,hide_index=True)
        st.markdown("**Data Dictionary**")
        dd = pd.DataFrame({
            "Kolom": ["Trip_Distance","Type_of_Cab","Customer_Since_Months","Life_Style_Index",
                      "Confidence_Life_Style_Index","Destination_Type","Customer_Rating",
                      "Cancellation_Last_1Month","Var1, Var2, Var3","Gender","Surge_Pricing_Type"],
            "Tipe": ["float64","object","float64","float64","object","object","float64",
                     "int64","float64","object","int64"],
            "Deskripsi": [
                "Jarak perjalanan yang diminta pelanggan",
                "Kategori taksi yang dipesan (A–E)",
                "Lama berlangganan dalam bulan; 0 = bulan berjalan",
                "Indeks gaya hidup proprietary Sigma Cabs berdasarkan perilaku pelanggan",
                "Kategori tingkat kepercayaan terhadap Life_Style_Index (A/B/C)",
                "Tipe destinasi dari 14 kategori (A–N)",
                "Rata-rata rating lifetime pelanggan",
                "Jumlah perjalanan yang dibatalkan dalam 1 bulan terakhir",
                "Variabel kontinu yang dimasking oleh perusahaan",
                "Jenis kelamin pelanggan",
                "Target — tipe surge pricing (1, 2, atau 3)"
            ]
        })
        st.dataframe(dd, use_container_width=True, hide_index=True)

    st.divider()

    # ════════════════════════
    # SECTION 2: UNIVARIATE
    # ════════════════════════
 

    st.markdown("<div class='section-header'>📊 Univariate Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d; font-size:14px;'>Eksplorasi distribusi fitur numerik dan kategori.</p>", unsafe_allow_html=True)

    # Gabungkan semua kolom (Numerik + Kategori)
    all_cols = df_clean.columns.tolist()

    # Dropdown tunggal untuk semua jenis variabel
    selected_var = st.selectbox("Pilih Variabel untuk Dianalisis", all_cols)

    # Panggil fungsi universal (hanya return satu fig)
    fig = get_univariate_plot(df_clean, selected_var)

    # Tampilkan Chart
    st.plotly_chart(fig, use_container_width=True)

    # Analysis Box dinamis
    st.markdown(f"""<div class='analysis-box'>
        Berdasarkan distribusi, <b>Var2</b> dan <b>Var3</b> menunjukkan distribusi yang skewed.
        Pada dasarnya perlu handling dengan transformasi, namun untuk keperluan DA ini dibiarkan dulu.
        Dari distribusi gender telihat pelanggan pria jauh lebih banyak dari wanita.
        Selain itu, karena target <b>Surge_Pricing_Type</b> hanya terbagi menjadi 3 tipe,
        variabel ini akan diubah sebagai kategori.
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ════════════════════════
    # SECTION 3: BIVARIATE
    # ════════════════════════
    st.markdown("<div class='section-header'>📈 Bivariate Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d; font-size:14px; margin-bottom:8px;'>Analisis hubungan tiap fitur terhadap Surge Pricing Type (Q1–Q8).</p>", unsafe_allow_html=True)

    q_options = {
        "Q1 · Apakah jarak perjalanan mempengaruhi surge pricing?": "Q1",
        "Q2 · Apakah tipe cab berpengaruh terhadap surge pricing?": "Q2",
        "Q3 · Apakah customer baru dan lama memiliki pola surge yang berbeda?": "Q3",
        "Q4 · Bagaimana lifestyle customer mempengaruhi harga?": "Q4",
        "Q5 · Apakah jenis destinasi mempengaruhi surge pricing?": "Q5",
        "Q6 · Bagaimana pengaruh rating customer terhadap harga?": "Q6",
        "Q7 · Bagaimana perilaku pembatalan mempengaruhi kenaikan harga?": "Q7",
        "Q8 · Apakah proporsi gender terhadap surge pricing setara?": "Q8",
    }
    selected_label = st.selectbox("Pilih Pertanyaan Analisis", list(q_options.keys()))
    selected_q = q_options[selected_label]

    bivariate_map = {
        "Q1": (def_num, {"col": "Trip_Distance",
               "title": "Q1: Analisis Jarak Tempuh vs Surge Pricing"}, False,
               """<b>Box Plot:</b> Median jarak perjalanan untuk Surge Type 1 dan Type 2 berada pada angka yang hampir serupa (~36 km).
               Perbedaan signifikan terlihat pada Surge Type 3, di mana median jaraknya melonjak ke angka <b>45 km</b>.
               Hal ini mengindikasikan bahwa perjalanan dengan jarak lebih jauh memiliki probabilitas lebih tinggi
               untuk masuk ke surge pricing tertinggi. Namun distribusi Type 1 dan Type 2 masih sangat overlap.<br><br>
               <b>Violin Plot:</b> Menunjukkan karakteristik multimodal pada ketiga tipe surge — mencerminkan rute-rute populer
               yang konsisten (~30, 45, 60, 75 km). Pada Type 3, terdapat penumpukan volume pada jarak ekstrem (~108 km)
               yang hampir tidak terlihat pada Type 1."""),
        "Q2": (def_cat, {"col": "Type_of_Cab",
               "title": "Q2: Distribusi Surge Pricing per Tipe Taksi (%)"}, False,
               """Data menunjukkan korelasi yang <b>sangat kuat</b> antara tipe kendaraan dengan tingkat surge pricing:<br><br>
               • <b>Tipe A (Standar):</b> 69% perjalanan pada Surge Type 1 — harga paling stabil.<br>
               • <b>Tipe B & C (Menengah):</b> Mayoritas pada Surge Type 2 (63% dan 61%).<br>
               • <b>Tipe D & E (Premium):</b> Didominasi Surge Type 3 — Tipe D sebesar 81%, Tipe E sebesar 72%.<br><br>
               Terdapat pola linier yang jelas: semakin tinggi kelas tipe kendaraan, semakin besar peluang
               terkena surge pricing level tertinggi. <b>Type_of_Cab merupakan variabel determinan utama.</b>"""),
        "Q3": (def_num, {"col": "Customer_Since_Months",
               "title": "Q3: Hubungan Masa Berlangganan Customer (Bulanan) Terhadap Surge Pricing"}, False,
               """<b>Box Plot:</b> Customer_Since_Months tidak menunjukkan korelasi yang signifikan terhadap tipe surge pricing.
               Ketiga kategori memiliki profil statistik yang <b>identik</b> — nilai median sama di angka 6 bulan,
               kuartil dan rentang min-max pun sama persis.<br><br>
               <b>Violin Plot:</b> Pola penumpukan data serupa di semua tipe, dengan puncak populasi tertinggi pada angka 10 bulan.
               Tidak ditemukan perbedaan antara pelanggan baru dan lama.
               Kesimpulan: <b>loyalitas pelanggan (durasi berlangganan) bukan variabel penentu dalam surge pricing.</b>"""),
        "Q4": (def_num, {"col": "Life_Style_Index",
               "title": "Q4: Pengaruh Gaya Hidup Customer Terhadap Surge Pricing"}, False,
               """<b>Box Plot:</b> Life_Style_Index memiliki nilai median yang <b>identik</b> di ketiga tipe surge pricing (~2.98).
               Profil gaya hidup rata-rata pelanggan Surge Type 1 maupun Type 3 tidak memiliki perbedaan signifikan.<br><br>
               <b>Violin Plot:</b> Lonjakan populasi terbesar berada tepat di angka 2.98 untuk semua kategori.
               Meskipun Q1 pada Surge Type 1 sedikit lebih tinggi, perbedaan ini terlalu kecil untuk dianggap konsisten.
               Kesimpulan: <b>Life_Style_Index bukan faktor penentu dalam penetapan Surge Pricing Type.</b>"""),
        "Q5": (def_cat, {"col": "Destination_Type", "sort_by_surge3": True,
               "title": "Q5: Perbandingan Surge Pricing antar Destinasi (%)<br>(Diurutkan dari Proporsi Surge 3 Tertinggi)"}, True,
               """Destinasi terbagi menjadi dua kelompok besar:<br><br>
               • <b>High Surge (Dominasi Type 3):</b> Mayoritas tipe destinasi menunjukkan proporsi Surge Type 3 tertinggi —
               mengindikasikan area dengan permintaan tinggi atau keterbatasan armada.<br>
               • <b>Moderate Surge (Dominasi Type 2):</b> Destinasi A, B, C, E, G, N memiliki Type 2 lebih dominan.
               Menariknya, destinasi awal (A, B, C) justru merupakan kontributor <b>volume transaksi terbesar</b>.<br><br>
               Tidak ada satu pun tipe destinasi yang didominasi Surge Type 1 —
               <b>harga normal sangat jarang terjadi di semua destinasi.</b>"""),
        "Q6": (def_num, {"col": "Customer_Rating",
               "title": "Q6: Efek Rating Customer terhadap Surge Pricing"}, False,
               """<b>Box Plot:</b> Customer Rating memiliki pengaruh nyata terhadap klasifikasi harga.
               Terdapat pola linier yang jelas: semakin rendah tipe surge (Type 1), semakin <b>tinggi</b> nilai median rating-nya.
               Sebaliknya, pada Surge Type 3, median rating berada di titik terendah.
               Ini mengonfirmasi adanya <b>korelasi negatif</b> antara kepuasan pelanggan dan tingkat kenaikan harga.<br><br>
               <b>Violin Plot:</b> Setiap tipe surge memiliki titik puncak kepadatan yang terkonsentrasi tepat
               di sekitar nilai median masing-masing — pola yang sangat konsisten."""),
        "Q7": (def_num, {"col": "Cancellation_Last_1Month",
               "title": "Q7: Dampak Histori Pembatalan terhadap Surge Pricing"}, False,
               """<b>Box Plot:</b> Untuk Surge Type 1 dan 2, nilai median masih di angka 0 — mayoritas transaksi
               dilakukan pelanggan tanpa riwayat pembatalan. Namun pada Surge Type 3, median naik menjadi <b>1</b>.
               Ini menunjukkan korelasi positif: semakin tinggi frekuensi pembatalan, semakin besar kemungkinan
               terkena surge tertinggi.<br><br>
               <b>Violin Plot:</b> Pada Type 1 dan 2, data sangat dominan di angka 0.
               Pada Type 3, volume di angka 0 berkurang dan menyebar ke angka lebih tinggi.
               Saat angka pembatalan ekstrem (≥3), <b>Type 3 mendominasi secara mutlak.</b>"""),
        "Q8": (def_cat, {"col": "Gender",
               "title": "Q8: Proporsi Gender Customer terhadap Surge Pricing (%)"}, False,
               """Distribusi surge pricing antara pelanggan <b>Female</b> dan <b>Male</b> menunjukkan proporsi yang
               hampir identik di ketiga tipe surge. Tidak ditemukan perbedaan yang signifikan antara kedua gender
               dalam hal paparan terhadap surge pricing.
               Kesimpulan: <b>Gender bukan faktor penentu dalam penetapan Surge Pricing Type.</b>"""),
    }

    fn, kwargs, _, analisis = bivariate_map[selected_q]
    fig = fn(df_clean, **kwargs)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<div class='analysis-box'>{analisis}</div>", unsafe_allow_html=True)

    st.divider()

    # ════════════════════════
    # SECTION 4: MULTIVARIATE
    # ════════════════════════
    st.markdown("<div class='section-header'>🗺️ Multivariate Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d; font-size:14px; margin-bottom:8px;'>Heatmap korelasi antar semua fitur dan target setelah encoding.</p>", unsafe_allow_html=True)

    st.plotly_chart(heatmap_corr_plotly(corr_matrix), use_container_width=True)

    st.markdown("<div class='section-header' style='font-size:16px; margin-top:8px;'>Fitur vs Target (Surge_Pricing_Type)</div>", unsafe_allow_html=True)
    st.markdown("""<div class='analysis-box'>
        <b>Type_of_Cab (0.50):</b> Penentu terkuat. Tipe taksi yang dipilih pelanggan punya pengaruh paling besar terhadap tipe surge.<br>
        <b>Cancellation_Last_1Month (0.19):</b> Pelanggan yang sering membatalkan perjalanan cenderung masuk ke surge yang lebih tinggi.<br>
        <b>Customer_Rating (-0.16):</b> Pelanggan dengan rating lebih baik cenderung mendapat surge type lebih rendah.<br>
        <b>Trip_Distance (0.14):</b> Semakin jauh perjalanan, semakin besar kemungkinan masuk surge lebih tinggi.<br>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='font-size:16px;'>Hubungan Antar Fitur</div>", unsafe_allow_html=True)
    st.markdown("""<div class='analysis-box'>
        <b>Var2 vs Var3 (0.68):</b> Hampir identik. Karena ini proyek DA tanpa modelling, sementara bisa dibiarkan dulu.<br>
        <b>Trip_Distance vs Life_Style_Index (0.47):</b> Ada keterkaitan moderat antara jarak perjalanan dan indeks gaya hidup pelanggan.<br>
        <b>Var2 vs Customer_Rating (-0.30):</b> Var2 bergerak berlawanan dengan rating pelanggan.<br>
        <b>Var3 vs Life_Style_Index (0.30):</b> Var3 punya keterkaitan dengan Life_Style_Index.<br>
        <b>Var3 vs Customer_Rating (-0.23):</b> Pola serupa dengan Var2 — Var3 juga cenderung berlawanan dengan rating pelanggan.<br>
        <b>Trip_Distance vs Var3 (0.23):</b> Jarak tempuh dan Var3 sedikit bergerak searah.<br>
        <b>Confidence_Life_Style_Index vs Trip_Distance (0.22):</b> Kategori kepercayaan data gaya hidup sedikit berkaitan dengan jarak perjalanan.<br>
        <b>Trip_Distance vs Destination_Type (-0.17):</b> Dari encoding A (0) hingga N (13), destinasi dengan kode lebih tinggi cenderung dikaitkan dengan perjalanan lebih pendek.<br>
        <b>Life_Style_Index vs Customer_Rating (0.19):</b> Pelanggan dengan Life_Style_Index lebih tinggi cenderung memiliki rating lebih baik.<br>
        <b>Destination_Type vs Customer_Rating (0.13):</b> Tipe destinasi sedikit memengaruhi rating pelanggan.<br>
        <b>Confidence_Life_Style_Index vs Var3 (0.11):</b> Ada hubungan positif yang lemah antara kategori kepercayaan data dan Var3.
    </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("<div class='section-header'>🔬 Uji Chi-Square</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#6c757d; font-size:14px;'>
    Tabel ini memvalidasi fitur kategorikal yang memiliki korelasi linear rendah namun memiliki 
    <b>asosiasi kuat</b> secara distribusi (P-Value < 0.05).
    </p>""", unsafe_allow_html=True)

    stat_data = [
        ("Type_of_Cab", "69,884.73", "0.0000", "✅ Sangat Kuat"),
        ("Destination_Type", "4,389.38", "0.0000", "✅ Signifikan*"),
        ("Confidence_Life_Style_Index", "4,709.39", "0.0000", "✅ Signifikan"),
        ("Gender", "2.67", "0.2629", "❌ Tidak Signifikan")
    ]
    stat_df = pd.DataFrame(stat_data, columns=["Fitur", "Chi2_Stat", "P-Value", "Kesimpulan"])
    
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

    # Note kecil di bawah tabel untuk memperkuat argumen Destination Type
    st.caption("*Catatan: Destination Type terbukti signifikan secara statistik memengaruhi Surge Type meskipun korelasinya rendah.")
    st.divider()

    # ════════════════════════
    # SECTION 5: INSIGHT
    # ════════════════════════
    st.markdown("<div class='section-header'>💡 Insight & Kesimpulan</div>", unsafe_allow_html=True)

    st.plotly_chart(corr_bar_plotly(corr_matrix), use_container_width=True)

    summary = [
        ("Q1", "Trip Distance", "✅ Berpengaruh", "Surge Type 3 memiliki median jarak lebih jauh (~45 km) dibanding Type 1 & 2 (~36 km)."),
        ("Q2", "Type of Cab", "✅ Berpengaruh Kuat", "Faktor determinan utama. Pola linier jelas: Tipe A → Surge 1, Tipe B/C → Surge 2, Tipe D/E → Surge 3."),
        ("Q3", "Customer Since Months", "❌ Tidak Berpengaruh", "Median, Q1, Q3 identik di ketiga tipe surge. Loyalitas pelanggan tidak relevan."),
        ("Q4", "Life Style Index", "❌ Tidak Berpengaruh", "Nilai median identik (~2.98) di semua tipe surge. Tidak ada pola yang membedakan."),
        ("Q5", "Destination Type", "✅ Berpengaruh (P-Value)", "Korelasi rendah, namun Chi-Square (p<0.05) membuktikan adanya pengaruh tipe destinasi pada Surge."),
        ("Q6", "Customer Rating", "✅ Berpengaruh", "Korelasi negatif konsisten: rating lebih tinggi → surge lebih rendah."),
        ("Q7", "Cancellation Last 1 Month", "✅ Berpengaruh", "Pelanggan dengan ≥3 pembatalan didominasi Surge Type 3."),
        ("Q8", "Gender", "❌ Tidak Berpengaruh", "Proporsi surge pricing hampir identik antara Female dan Male."),
    ]
    summary_df = pd.DataFrame(summary, columns=["Kode", "Fitur", "Pengaruh", "Insight"])
    def color_influence(val):
        if "✅" in val: return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        if "❌" in val: return 'background-color: #f8d7da; color: #721c24;'
        return ''

    st.dataframe(
    summary_df.style.map(color_influence, subset=['Pengaruh']), 
    width="stretch", 
    hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""<div class='card' style='border-top: 3px solid #636EFA;'>
            <div class='card-label'>Strategi Pricing</div>
            <div class='card-title'>Segmentasi Berbasis Tipe Cab</div>
            <div class='card-body'>Gunakan Type_of_Cab sebagai basis utama segmentasi harga. Transparansi surge per tipe kendaraan dapat meningkatkan kepercayaan pelanggan premium.</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""<div class='card' style='border-top: 3px solid #EF553B;'>
            <div class='card-label'>Operasional</div>
            <div class='card-title'>Monitor Perilaku Pembatalan</div>
            <div class='card-body'>Pelanggan dengan riwayat pembatalan tinggi berkorelasi dengan Surge Type 3. Sistem peringatan dini dapat membantu optimasi supply-demand.</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""<div class='card' style='border-top: 3px solid #00CC96;'>
            <div class='card-label'>Customer Experience</div>
            <div class='card-title'>Program Loyalitas Berbasis Rating</div>
            <div class='card-body'>Pelanggan dengan rating tinggi cenderung mendapat surge lebih rendah. Program reward untuk pelanggan berperilaku baik dapat mendorong kepuasan dan retention.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='analysis-box' style='font-size:15px; line-height:2; margin-top:20px;'>
        Dari 8 analytical questions yang dianalisis, ditemukan bahwa <b>5 faktor</b> memiliki pengaruh nyata terhadap Surge Pricing Type:
        <b>Type_of_Cab</b> (dominan, r=0.50), <b>Cancellation_Last_1Month</b> (r=0.19),
        <b>Customer_Rating</b> (r=-0.16), <b>Trip_Distance</b> (r=0.14), dan <b>Destination_Type</b> (r kecil tapi p-value < 0.05).
        Sementara <b>Customer_Since_Months</b>, <b>Life_Style_Index</b>, dan <b>Gender</b>
        tidak menunjukkan perbedaan yang berarti antar tipe surge.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# DATA SCIENCE — PLACEHOLDER
# ═════════════════════════════════════════════
elif role == "Data Science":
    st.markdown("""<div class='coming-soon'>
        <span>🔬</span>
        Data Science track belum tersedia.<br>Coming soon.
    </div>""", unsafe_allow_html=True)