import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import remote_ai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remote_ai import get_feedback_from_ai

class TestRemoteAI(unittest.TestCase):
    
    @patch('remote_ai.query')
    @patch.dict(os.environ, {'HF_TOKEN': 'mock_token'})
    def test_feedback_success(self, mock_query):
        # Mock successful response
        mock_query.return_value = [{'generated_text': 'HRV Analysis: Good.'}]
        
        metrics = {'rmssd': 50, 'lf_hf_ratio': 1.5, 'sdnn': 40}
        result = get_feedback_from_ai(metrics)
        self.assertEqual(result, 'HRV Analysis: Good.')

    @patch('remote_ai.query')
    @patch.dict(os.environ, {'HF_TOKEN': 'mock_token'})
    def test_feedback_fallback(self, mock_query):
        # Mock primary failure then fallback success
        # First call fails
        mock_query.side_effect = [
            {'error': 'Model loading'}, # Primary fails
            [{'generated_text': 'Fallback Analysis'}] # Fallback succeeds
        ]
        
        metrics = {'rmssd': 50}
        result = get_feedback_from_ai(metrics)
        self.assertEqual(result, 'Fallback Analysis')

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_token(self):
        result = get_feedback_from_ai({})
        self.assertIn("Error: Hugging Face API Token (HF_TOKEN) is missing", result)

if __name__ == '__main__':
    unittest.main()
