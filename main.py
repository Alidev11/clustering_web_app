import streamlit as st


def stream_app():
    # Use a breakpoint in the code line below to debug your script.
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from sklearn.cluster import DBSCAN
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    st.set_page_config(page_title='California Houses Clustering',
                       page_icon="🧊")

    @st.experimental_memo
    def load_data():
        data = pd.read_csv('california_housing_train.csv')
        return data

    df = load_data()
    lat_long = df[['latitude', 'longitude']]
    X = lat_long.to_numpy()
    fig = px.scatter(lat_long, x="longitude", y="latitude")
    dbscan_cluster_model = DBSCAN(eps=0.38421052631578945, min_samples=26).fit(X)
    lat_long['cluster'] = dbscan_cluster_model.labels_
    fig1 = px.scatter(lat_long, x="longitude", y="latitude", color=lat_long['cluster'])

    with st.sidebar:
        st.title("California Houses Clustering")
        page = st.radio("Navigation bar", ["Data Exploration", "Clustering", "Infos"])

    st.markdown(
        f"<hr/>",
        unsafe_allow_html=True)
    st.markdown(
        f"<h1 style='text-align: center; text-transform:capitalize; font-size: 3rem; display: flex; align-items: center; justify-content: center; font-size: 3rem;'>California Houses Clustering</h1>",
        unsafe_allow_html=True)
    st.markdown(
        f"<hr/>",
        unsafe_allow_html=True)

    if page == "Data Exploration":
        st.markdown("#### Data Exploration - Dataset Numbers")
        num_class, shape_dim = st.columns(2)
        with num_class:
            st.markdown("**Number of clusters**")
            number1 = len(pd.unique(lat_long['cluster'])) - 1
            st.markdown(f"<h3 style='text-align: center; color: red;'>{number1}</h3>", unsafe_allow_html=True)

        with shape_dim:
            st.markdown("**Shape of dataset**")
            number2 = df.shape
            st.markdown(f"<h3 style='text-align: center; color: red;'>{number2}</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<hr/>",
            unsafe_allow_html=True)
        st.markdown("**Dataset**")
        st.dataframe(df, use_container_width=True)
        st.markdown("**Dataset description**")
        st.dataframe(df.describe(), use_container_width=True)
        fig4 = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, vertical_spacing=0.2,
                             subplot_titles=('Longitude frequency', 'Latitude frequency'),
                             column_widths=[0.3, 0.3], row_heights=[0.1])
        fig5 = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, vertical_spacing=0.2,
                             subplot_titles=('housing_median_age frequency', 'total_rooms frequency'),
                             column_widths=[0.3, 0.3], row_heights=[0.1])
        fig6 = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, vertical_spacing=0.2,
                             subplot_titles=('total_bedrooms frequency', 'population frequency'),
                             column_widths=[0.3, 0.3], row_heights=[0.1])
        fig7 = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, vertical_spacing=0.2,
                             subplot_titles=('households frequency', 'median_income frequency'),
                             column_widths=[0.3, 0.3], row_heights=[0.1])
        fig8 = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, vertical_spacing=0.2,
                             subplot_titles=('median_house_value frequency', 'median_income frequency'),
                             column_widths=[0.3, 0.3], row_heights=[0.1])

        trace0 = go.Histogram(x=df["longitude"], nbinsx=30, name="longitude", showlegend=True)
        trace1 = go.Histogram(x=df["latitude"], nbinsx=30, name="latitude", showlegend=True)


        fig4.append_trace(trace0, 1, 1)
        fig4.append_trace(trace1, 1, 2)
        fig4.update_layout(
            title_text='Data Distribution',  # title of plot
            bargap=0,  # gap between bars of adjacent location coordinates
            bargroupgap=0  # gap between bars of the same location coordinates
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ------------------------------------------------- Clustering plot1 -----------------------------------------
    if page == "Clustering":
        st.markdown(
            f"<h1 style='text-align: center; text-transform:capitalize; height: 8vh; display: flex; align-items: center; justify-content: center; font-size:1.8rem;'>Before clustering</h1>",
            unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        # ------------------------------------- After Clustering plot2 ---------------------------------------------
        st.markdown(
            f"<h1 style='text-align: center; text-transform:capitalize; height: 8vh; display: flex; align-items: center; justify-content: center; font-size:1.8rem;'>After clustering</h1>",
            unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)

        # ----------------------------------------- Subplot after cluster ----------------------------------------
        labels = ["Cluster1", "Cluster2", "Cluster3", "Outliers"]
        fig10 = go.Figure(data=[go.Pie(labels=labels, values=[
            lat_long[lat_long["cluster"] == 0].count().cluster,
            lat_long[lat_long["cluster"] == 1].count().cluster,
            lat_long[lat_long["cluster"] == 2].count().cluster,
            lat_long[lat_long["cluster"] == -1].count().cluster],
                                       textinfo='label+percent',
                                       insidetextorientation='radial'
                                       )])
        st.plotly_chart(fig10, use_container_width=True)
    if page == "Infos":
        st.markdown(
            f"<p style='text-align: left; font-size:1rem;'>This a web application that serves as a dashboard for the California houses dataset. It contains two main parts: Data Exploration and Clustering.</p>",
            unsafe_allow_html=True)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    stream_app()
