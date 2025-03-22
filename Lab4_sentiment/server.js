const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const path = require('path');
const cors = require('cors');
const app = express();
const port = process.env.PORT || 3000;

const HF_API_TOKEN = "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";
const HF_API_URL = 'https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english';
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

app.post('/sentiment', async (req, res) => {
    const { text } = req.body;
    if (!text) {
        return res.status(400).json({ error: 'No text provided' });
    }
    try {
        const response = await axios.post(
            HF_API_URL,
            { inputs: text },
            {
                headers: {
                    'Authorization': `Bearer ${HF_API_TOKEN}`,
                    'Content-Type': 'application/json'
                }
            }
        );
        let result = response.data;
        if (Array.isArray(result) && result.length > 0) {
            result = result.reduce((max, current) => current.score > max.score ? current : max, result[0]);
        }
        res.json({ result });
    } catch (error) {
        res.status(error.response ? error.response.status : 500).json({
            error: 'Error calling Hugging Face API',
            details: error.response ? error.response.data : error.message
        });
    }
});
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
