import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestChatEndpoint:
    """Test the POST /chat endpoint"""

    @patch("main.get_ai_response", new_callable=AsyncMock)
    def test_chat_returns_valid_response(self, mock_ai):
        """
        Given a valid message, /chat returns 200 with reply and sources.
        We mock the Groq call — don't want to call the real API during tests.
        """
        # Setup: mock the AI to return a fixed response
        mock_ai.return_value = "Mocked AI response"
        
        # Act: send a message
        response = client.post(
            "/chat",
            json={"message": "What is your refund policy?"}
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == "Mocked AI response"
        assert isinstance(data["sources"], list)

    def test_chat_missing_message_field(self):
        """POST /chat without 'message' field returns 422"""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_empty_message(self):
        """POST /chat with empty message should either reject or handle gracefully"""
        response = client.post("/chat", json={"message": ""})
        # You decide: should this be 400 or processed?
        # For now, just document the behaviour:
        assert response.status_code == 400

    @patch("main.get_ai_response", new_callable=AsyncMock)
    def test_chat_handles_groq_error(self, mock_ai):
        """If Groq fails, /chat returns 500 (or a user-friendly error)"""
        mock_ai.side_effect = Exception("Groq API is down")
        
        response = client.post("/chat", json={"message": "test"})
        
        # Your API should return 500, not crash
        assert response.status_code == 500