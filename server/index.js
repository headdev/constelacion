const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const port = 3000;

// Use CORS middleware
app.use(cors());

app.get('/prediction', (req, res) => {
    const filePath = path.join(__dirname, '..', 'data', 'prediction.json');

    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading the file', err);
            return res.status(500).json({ error: 'Internal Server Error' });
        }

        try {
            const jsonData = JSON.parse(data);
            res.json(jsonData);
        } catch (err) {
            console.error('Error parsing JSON', err);
            res.status(500).json({ error: 'Internal Server Error' });
        }
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
