import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [prediction, setPrediction] = useState('');
    const [imagePreview, setImagePreview] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);

        const reader = new FileReader();
        reader.onloadend = () => {
            setImagePreview(reader.result);
        };
        reader.readAsDataURL(file);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error('There was an error!', error);
        }
    };

    return (
        <div className="App">
            <h1>Image Classification</h1>
            <form onSubmit={handleSubmit}>
            <label htmlFor="file-upload" className="custom-file-upload">
              Choose File
            </label>
            <input id="file-upload" type="file" onChange={handleFileChange} />                <button type="submit">Upload and Predict</button>
            </form>
            {imagePreview && (
                <div>
                    <h2>Selected Image:</h2>
                    <img src={imagePreview} alt="Selected" style={{ width: '300px' }} />
                </div>
            )}
            {prediction && <h2>Prediction: {prediction}</h2>}
        </div>
    );
}

export default App;
