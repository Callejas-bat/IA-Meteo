const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Configuración del servidor
const app = express();
const port = process.env.PORT || 3000;

let model, scaler, labelEncoder;

// Función para cargar archivos JSON
const loadJSONFile = (filePath) => {
    try {
        const fullPath = path.join(__dirname, 'public', filePath);
        return JSON.parse(fs.readFileSync(fullPath, 'utf8'));
    } catch (error) {
        console.error(`Error loading JSON file ${filePath}:`, error);
        throw error;
    }
};

// Cargar el modelo, el scaler y el label encoder al iniciar el servidor
(async () => {
    try {
        // URL del modelo y otros recursos
        const modelPath = path.join(__dirname, 'public', 'model.json');

        // Cargar el modelo
        model = await tf.loadGraphModel(`file://${modelPath}`);
        console.log('Modelo cargado');

        // Cargar el scaler y el label encoder
        scaler = loadJSONFile('scaler.json');
        labelEncoder = loadJSONFile('label_encoder.json');

        console.log('Scaler:', scaler);
        console.log('Label Encoder:', labelEncoder);

    } catch (error) {
        console.error('Error al cargar el modelo o los archivos auxiliares:', error);
    }
})();

// Middleware para manejar datos en formato JSON
app.use(express.json());

// Ruta principal para predicciones
app.get('/predict', async (req, res) => {
    try {
        const { hum, luz, pres, temp, vel } = req.query;

        if ([hum, luz, pres, temp, vel].some(val => isNaN(parseFloat(val)))) {
            return res.status(400).json({ error: 'Invalid input parameters. Ensure all parameters are numbers.' });
        }

        if (!scaler || !labelEncoder) {
            return res.status(500).json({ error: 'Scaler or label encoder not loaded' });
        }

        const input = [parseFloat(hum), parseFloat(luz), parseFloat(pres), parseFloat(temp), parseFloat(vel)];
        const scaledInput = input.map((val, i) => {
            if (!scaler.mean || !scaler.scale) {
                throw new Error('Scaler.mean or scaler.scale is undefined');
            }
            return (val - scaler.mean[i]) / scaler.scale[i];
        });

        const tensorInput = tf.tensor2d([scaledInput], [1, 5]);
        const prediction = model.predict(tensorInput);
        const predClass = prediction.argMax(-1).dataSync()[0];
        const result = labelEncoder.classes[predClass];

        return res.status(200).json({ prediction: result });

    } catch (error) {
        console.error('Error during prediction:', error);
        return res.status(500).json({ error: `Error during prediction: ${error.message}` });
    }
});

app.listen(port, () => {
    console.log(`Servidor corriendo en http://localhost:${port}`);
});
