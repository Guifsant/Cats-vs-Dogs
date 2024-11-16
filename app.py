import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import base64
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Configurações do aplicativo Dash com Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Carregar o modelo treinado
model = load_model('model_cats_vs_dogs.h5')

# Função para preparar a imagem antes de fazer a predição
IMG_SIZE = 128  # Tamanho da imagem usado durante o treinamento

def prepare_image(image_path):
    # Carregar e redimensionar a imagem
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    # Converter para um array numpy
    img_array = img_to_array(img)
    # Adicionar uma dimensão extra (batch size = 1)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalizar (se foi feito rescale no treinamento)
    img_array = img_array / 255.0
    return img_array

# Layout do aplicativo com Dash Bootstrap
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Classificador de Gato ou Cachorro", className="text-center"),
                width=12,
            ),
            className="mb-4"
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    dcc.Upload(
                        id='upload-image',
                        children=html.Button('Upload Imagem', className="btn btn-primary"),
                        accept='image/*',
                        style={'width': '100%'}
                    ),
                    width=12, 
                    className="text-center",
                ),
            ],
            className="mb-4"
        ),
        
        dbc.Row(
            [
                # Centralizar as colunas horizontalmente usando d-flex e justify-content-center
                dbc.Col(
                    html.Div(id="output-image-upload", className="text-center"),
                    width=6,
                    className="d-flex justify-content-center align-items-center"
                ),
                dbc.Col(
                    html.Div(id="prediction-result", className="text-center"),
                    width=6,
                    className="d-flex justify-content-center align-items-center"
                ),
            ],
            className="mb-4 d-flex justify-content-center align-items-center",  # Centraliza as colunas horizontalmente
        ),
    ],
    fluid=True,
    className="d-flex flex-column justify-content-center align-items-center",
    style={'minHeight': '100vh'},
)

# Função callback para lidar com o upload da imagem e fazer a predição
@app.callback(
    [Output("output-image-upload", "children"),
     Output("prediction-result", "children")],
    [Input("upload-image", "contents")]
)
def update_output(contents):
    if contents is None:
        return '', ''

    # Decodificar a imagem recebida em base64
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Salvar a imagem recebida
    image_path = os.path.join("uploads", "uploaded_image.jpg")
    with open(image_path, 'wb') as f:
        f.write(decoded)
    
    # Preparar a imagem para o modelo
    image_array = prepare_image(image_path)

    # Realizar a predição
    prediction = model.predict(image_array)

    # Exibir o resultado
    if prediction[0] > 0.5:
        result = f"A imagem é um cachorro (Confiança: {prediction[0][0]*100:.2f}%)"
    else:
        result = f"A imagem é um gato (Confiança: {(1 - prediction[0][0])*100:.2f}%)"
    
    # Exibir a imagem no app
    image_component = html.Div([
        html.H5(f"Imagem carregada:"),
        html.Img(src=contents, style={'width': '300px', 'height': '300px', 'margin': '0 auto'}),
    ])
    
    return image_component, result

if __name__ == '__main__':
    app.run_server(debug=True)
