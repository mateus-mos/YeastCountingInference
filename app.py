import streamlit as st
from PIL import Image, ImageDraw
import tempfile
from inference_sdk import InferenceHTTPClient

# --- Replace with your Roboflow API key and model details ---
ROBOFLOW_API_KEY = "BjCE4IQzwn9VFOGPR9En"
ROBOFLOW_MODEL_ID = "yeast-cell-counting-v2-ao81v/3"

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

def infer_image(image_file):
    """Envia a imagem para o SDK de Inferência da Roboflow usando um arquivo temporário."""
    try:
        # Cria um arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            # Escreve o conteúdo do arquivo carregado no arquivo temporário
            image_bytes = image_file.read()
            tmp_file.write(image_bytes)
            temp_file_path = tmp_file.name
            print(f"Caminho do arquivo temporário em infer_image: {temp_file_path}") # Debugging

        # Realiza a inferência na imagem local usando o caminho do arquivo
        result = CLIENT.infer(temp_file_path, model_id=ROBOFLOW_MODEL_ID)

        # Limpa o arquivo temporário
        import os
        os.remove(temp_file_path)
        print(f"Arquivo temporário removido: {temp_file_path}") # Debugging

        return result
    except Exception as e:
        st.error(f"Erro durante a inferência: {e}")
        return None

def main():
    st.title("Testador do Modelo de Contagem de Células de Levedura")
    st.write("Carregue uma imagem para testar o modelo de contagem de células de levedura.")

    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            st.write("### Processando...")

            # Realiza a inferência
            results = infer_image(uploaded_file) # Passa o arquivo da imagem

            st.write("### Imagem Carregada")
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem Carregada.", use_container_width=True) # Changed here

            draw = ImageDraw.Draw(image)
            image_width, image_height = image.size

            # Define um dicionário para mapear nomes de classes para cores
            class_colors = {
                "budding-box": "red",
                "budding-cell": "blue",
                "cell": "green",
                # Adicione mais classes e cores conforme necessário
            }

            class_counts = {}
            total_cells = 0

            if results and 'predictions' in results: # Verifica se a chave 'predictions' existe
                predictions = results['predictions']

                for prediction in predictions:
                    class_name = prediction.get('class') # Acessa 'class' usando get
                    confidence = prediction.get('confidence') # Acessa 'confidence' usando get
                    x = prediction.get('x')
                    y = prediction.get('y')
                    width = prediction.get('width')
                    height = prediction.get('height')

                    # Conta as ocorrências de cada classe
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    # Calcula o total de células (excluindo budding-box)
                    if class_name in ["cell", "budding-cell"]:
                        total_cells += 1

                    # Calcula as coordenadas da caixa delimitadora
                    left = int(x - (width / 2))
                    top = int(y - (height / 2))
                    right = int(x + (width / 2))
                    bottom = int(y + (height / 2))

                    # Obtém a cor para a classe atual, padrão para preto se não encontrado
                    box_color = class_colors.get(class_name, "black")
                    text_color = box_color

                    # Desenha a caixa delimitadora
                    draw.rectangle([(left, top), (right, bottom)], outline=box_color, width=3)

                    # Adiciona o rótulo com o nome da classe e a confiança
                    text = f"{class_name}: {confidence:.2f}"
                    draw.text((left, top - 10), text, fill=text_color)

                st.write("### Resultado da Detecção")
                st.image(image, caption="Imagem com Caixas Delimitadoras.", use_container_width=True) # Changed here

                st.write("### Sumário da Detecção")
                st.write(f"**Total de Células (célula + budding-cell):** {total_cells}")
                st.write("**Número de elementos por classe:**")
                for class_name, count in class_counts.items():
                    st.write(f"- **{class_name}:** {count}")

            elif results:
                st.write("Nenhuma previsão encontrada na resposta da API.")

        except Exception as e:
            st.error(f"Erro durante o processamento em main: {e}")

if __name__ == "__main__":
    main()