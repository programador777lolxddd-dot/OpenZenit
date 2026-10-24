import unittest
from mini_codegpt import Tokenizer

class TokenizerTests(unittest.TestCase):
    def test_round_trip(self):
        text = 'Hola Zenit!'
        tokenizer = Tokenizer(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

if __name__ == '__main__':
    unittest.main()
