import json
import os
from pathlib import Path

import requests
from responses import PageSpeedResponse


class PageSpeed(object):
    """
    Google PageSpeed analysis client

    Attributes:
        api_key (str): Optional API key for client account.
        endpoint (str): Endpoint for HTTP request
    """

    def __init__(self, api_key: str | None = None, timeout: float = 30) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.timeout = timeout
        self.endpoint = 'https://www.googleapis.com/pagespeedonline/v5/runPagespeed'

    def analyse(self, url, strategy='desktop', category='performance'):
        """
        Run PageSpeed test

        Args:
            url (str): The URL to fetch and analyse.
            strategy (str, optional): The analysis strategy to use. Acceptable values: 'desktop', 'mobile'
            category (str, optional): A Lighthouse category to run; if none are given, only Performance category will be run

        Returns:
            response: PageSpeed API results
        """
        strategy = strategy.lower()

        if strategy not in ('mobile', 'desktop'):
            raise ValueError('invalid strategy: {0}'.format(strategy))
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        params = {
            'strategy': strategy,
            'url': url,
            'category': category,
        }

        if self.api_key:
            params['key'] = self.api_key

        raw = requests.get(self.endpoint, params=params, timeout=self.timeout)

        response = PageSpeedResponse(raw)

        return response

    def save(self, response: PageSpeedResponse, path: str | Path = "json_data.json") -> Path:
        json_data = response._json
        output = Path(path)
        if output.is_dir():
            output = output / "json_data.json"
        output.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
        return output
