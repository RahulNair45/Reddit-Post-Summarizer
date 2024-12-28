from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def click_show_more_button(driver):
    """
    Attempts to click the 'Show More' button if it exists.
    """

    try:
        show_more_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Show more')]")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_more_button)
        time.sleep(1)  
        driver.execute_script("arguments[0].click();", show_more_button)
        print("Successfully clicked 'Show More' button.")
    except Exception as e: 
        print(f"No 'Show More' button found or could not click: {e}") 


def scrape_multiplayer_games(max_games):
    print(f"Scraping up to {max_games} multiplayer games from Steam...")
    games = []
    driver = webdriver.Chrome()  
    driver.get("https://store.steampowered.com/category/multiplayer/?flavor=contenthub_all&facets13268=6%3A4")
    time.sleep(10)  

    try:
        while len(games) < max_games:
            time.sleep(3)

            game_elements = driver.find_elements(By.CLASS_NAME, "_2ekpT6PjwtcFaT4jLQehUK.StoreSaleWidgetTitle")
            for game in game_elements:
                game_title = game.text.strip()
                if game_title not in games:
                    games.append(game_title) 
                    print(f"Scraped: {game_title}")

                if len(games) >= max_games:
                    break

            print(f"Scraped {len(game_elements)} games on this page. Total so far: {len(games)}")

            click_show_more_button(driver)

    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        driver.quit()

    return games

if __name__ == "__main__":
    max_games = 10000
    games = scrape_multiplayer_games(max_games)

    print("\nScraped Multiplayer Games:")
    for idx, game in enumerate(games, start=1):
        print(f"{idx}: {game}")

    with open("multiplayer_games_max.txt", "w", encoding="utf-8") as file:
        for game in games:
            file.write(game + "\n")
    print(f"\nGame titles saved to 'multiplayer_games_limited.txt'.")