// app.js
document.addEventListener('DOMContentLoaded', () => {
    fetch('http://localhost:3000/prediction')
        .then(response => response.json())
        .then(data => {
            console.log(data);
            const tableBody = document.querySelector('#signalsTable tbody');
            tableBody.innerHTML = ''; // Limpiar la tabla antes de agregar nuevas filas

            // Crear una nueva fila para la señal recibida
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${data.token}</td>
                <td>${data.predicted_price}</td>
                <td>${data.prediction_date}</td>
                <td>${data.prediction_days}</td>
            `;
            tableBody.appendChild(row);
        })
        .catch(error => console.error('Error al obtener las señales:', error));
});
