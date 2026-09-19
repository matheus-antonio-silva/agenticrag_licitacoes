from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> list[dict]:

    results = []

    with DDGS() as ddgs:
        for item in ddgs.text(
            query,
            max_results=max_results
        ):
            results.append(
                {
                    "title": item.get("title"),
                    "href": item.get("href"),
                    "body": item.get("body"),
                }
            )

    return results


if __name__ == "__main__":

    results = search_web(
        "Quem ganhou a Copa do Mundo de 2022?"
    )

    for index, result in enumerate(
        results,
        start=1
    ):
        print(f"\n--- Resultado {index} ---")
        print(result["title"])
        print(result["body"])
        print(result["href"])