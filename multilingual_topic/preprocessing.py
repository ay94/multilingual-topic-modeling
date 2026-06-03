import re
import unittest
from emoji import replace_emoji

class TextPreprocessor():
    def __init__(self,
                 apply_remove_links=True,
                 apply_remove_mentions=True,
                 apply_remove_hashtags=True,
                 apply_remove_emojis=True,
                 apply_remove_spaces=True,
                 apply_arabic_preprocessor=False,
                 ):
        self.apply_remove_links = apply_remove_links
        self.apply_remove_mentions = apply_remove_mentions
        self.apply_remove_hashtags = apply_remove_hashtags
        self.apply_remove_emojis = apply_remove_emojis
        self.apply_remove_spaces = apply_remove_spaces
        self.apply_arabic_preprocessor = apply_arabic_preprocessor

        if self.apply_arabic_preprocessor:
            from arabert.preprocess import ArabertPreprocessor
            self.arabic_preprocessor = ArabertPreprocessor('aubmindlab/bert-base-arabertv02')
        else:
            self.arabic_preprocessor = None

    def _remove_links(self, text):

        clean_txt = re.sub(r'http\S+', '', text)  # Removes links starting with http
        clean_txt = re.sub(r'www\.\S+', '', clean_txt)  # Removes links starting with www.
        clean_txt = re.sub(r'\(www\.\S+\)', '', clean_txt)  # Removes links in parentheses
        return re.sub(r'www.\S+', '', clean_txt)

    def _remove_mentions(self, text):
        return re.sub("@[a-zA-Z0-9_]+", r'', text)

    def _remove_hashtags(self, text):
        return text.replace("#", '')

    def _replace_emojis(self, text):
        # Unicode ranges for emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"  # symbols
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   "]+", flags=re.UNICODE)

        return emoji_pattern.sub(r'', text)

    def _remove_emojis(self, text):
        text = replace_emoji(text, ' ')
        text = text.strip()
        return text

    def _normalise_spaces(self, text):
        # Replace the repeated space token with the *same* space token that was repeated
        text = re.sub(r'(\s)\1+', r'\1', text)
        text = text.strip()
        return text

    def preprocess(self, text):
        clean_txt = text

        if self.apply_remove_links:
            clean_txt = self._remove_links(clean_txt)
        if self.apply_remove_emojis:
            clean_txt = self._remove_emojis(clean_txt)
        if self.apply_remove_mentions:
            clean_txt = self._remove_mentions(clean_txt)
        if self.apply_remove_hashtags:
            clean_txt = self._remove_hashtags(clean_txt)
        if self.apply_remove_spaces:
          clean_txt = self._normalise_spaces(clean_txt)
        if self.apply_arabic_preprocessor:
            clean_txt = self.arabic_preprocessor.preprocess(clean_txt)

        return clean_txt




class TestTextPreprocessor(unittest.TestCase):

    def test_remove_links(self):
        preprocessor = TextPreprocessor(apply_remove_links=True)
        self.assertEqual(preprocessor.preprocess("Check this link http://example.com"), "Check this link")
        self.assertEqual(preprocessor.preprocess("Visit www.example.com for more info"), "Visit for more info")
        self.assertEqual(preprocessor.preprocess("Link in parentheses (www.example.com)"), "Link in parentheses (")

    def test_remove_mentions(self):
        preprocessor = TextPreprocessor(apply_remove_mentions=True)
        self.assertEqual(preprocessor.preprocess("Hello @user!"), "Hello !")

    def test_remove_hashtags(self):
        preprocessor = TextPreprocessor(apply_remove_hashtags=True)
        self.assertEqual(preprocessor.preprocess("#testing this hashtag"), "testing this hashtag")

    def test_remove_emojis(self):
        preprocessor = TextPreprocessor(apply_remove_emojis=True)
        # Add an appropriate test for emojis

    def test_normalise_spaces(self):
        preprocessor = TextPreprocessor(apply_remove_spaces=True)
        self.assertEqual(preprocessor.preprocess("Too  many   spaces"), "Too many spaces")

    # Add more tests for other methods and combinations


def interactive_preprocessor_test(text):
    preprocessor = TextPreprocessor(
        apply_remove_links=True,
        apply_remove_mentions=True,
        apply_remove_hashtags=True,
        apply_remove_emojis=True,
        apply_remove_spaces=True
    )
    processed_text = preprocessor.preprocess(text)
    print("Original Text:", text)
    print("Processed Text:", processed_text)
