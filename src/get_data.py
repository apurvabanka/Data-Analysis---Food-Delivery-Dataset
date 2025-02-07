import pandas as pd

from bs4 import BeautifulSoup

from utils import load_selenium


def load_website(progress_bar, status_text):
    df_list = []
    url_list = [
        "https://ritik1129.github.io/Food_Delivery_Dataset/",
        "https://ritik1129.github.io/Food_Delivery_Dataset/order_details_table",
        "https://ritik1129.github.io/Food_Delivery_Dataset/location_details_table"
    ]

    current_progress = 10

    progress_bar.progress(current_progress)

    print("################################################")
    print("Starting to Laod Website")
    status_text.text("Starting to Laod Website")
    print("################################################")


    for url in url_list:
        print("################################################")
        print("Processing URL - ", url)
        status_text.text("Processing URL - " + url)
        print("################################################")

        driver = load_selenium(url)
        html = driver.page_source

        soup = BeautifulSoup(html, 'html.parser')

        table = soup.find('table', {'id': 'csvTable'})
        if table:
            headers = [header.text.strip() for header in table.find_all('th')]

            rows = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = [cell.text.strip() for cell in row.find_all('td')]
                rows.append(cells)

            df = pd.DataFrame(rows, columns=headers)
        else:
            print("No table found on the page.")

        driver.quit()
        df_list.append(df)

        current_progress += 25
        progress_bar.progress(current_progress)
    
    status_text.text("Web Scrapping Successfully")

    return df_list

def combine_df(df_list):
    df = df_list[0].merge(df_list[1], on='ID', how='outer').merge(df_list[2], on='ID', how='outer')
    df = df[1:]

    return df
