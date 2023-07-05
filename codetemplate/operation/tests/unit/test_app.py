import requests

import unittest
from unittest.mock import patch

from codetemplate.webapp import app

url = 'https://www.gfk.com/home'


def url_exists(url):
    r = requests.get(url)
    if r.status_code == 200:
        return True

    elif r.status_code == 404:
        return False


class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

    def test_online_data_received(self):
        """
        Test to check whether the data has been received or not
        :return:
        """
        res = requests.get(url)
        print(res.status_code)
        self.assertEqual(200, res.status_code)

    def test_invalid_online_doc_received(self):
        """
        Test whether the data source is invalid
        """
        res = requests.get(url + 'fake')
        self.assertGreater(res.status_code, 400)

    def test_data_content_not_null(self):
        """
        Test whether the data has null content
        :return:
        """
        res = requests.get(url)
        self.assertIsNotNone(res.text)

    def test_returns_true_if_url_found(self):
        """
        Test whether the url exists
        :return:
        """
        with patch('requests.get') as mock_request:

            # set a `status_code` attribute on the mock object
            # with value 200
            mock_request.return_value.status_code = 200

            self.assertTrue(url_exists(url))

            # test if requests.get was called
            # with the given url or not
            mock_request.assert_called_once_with(url)

    def test_returns_false_if_url_not_found(self):
        """
        Test the url does not exist
        :return:
        """
        with patch('requests.get') as mock_request:

            # set a `status_code` attribute on the mock object
            # with value 404
            mock_request.return_value.status_code = 404

            self.assertFalse(url_exists(url))

            # test if requests.get was called
            # with the given url or not
            mock_request.assert_called_once_with(url)


if __name__ == '__main__':
    unittest.main()
