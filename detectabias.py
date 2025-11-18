import os
import re
import json
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv 
import google.generativeai as genai
import streamlit.components.v1 as components # Import para a função de PDF

# === ALTERNATIVA DEFINITIVA PARA EXTRAÇÃO DE PDF (PyMuPDF) ===
try:
    import fitz # PyMuPDF
    PDF_LIB_AVAILABLE = True
except ImportError:
    PDF_LIB_AVAILABLE = False
    st.error("ERRO: O PyMuPDF (biblioteca 'fitz') não está instalado. Por favor, rode 'pip install PyMuPDF' no seu Venv.")
    st.stop()
# ==============================================================

# === REMOVIDO ===
# Bloco de importação do Google Sheets (gspread) removido.
# =================

# === CONFIGURAÇÕES INICIAIS ===
st.set_page_config(page_title="DetectaBias IA", layout="wide")
load_dotenv() 

# Configuração da API Gemini: Lendo a chave diretamente do Streamlit Secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    GEMINI_AVAILABLE = True
except KeyError:
    GEMINI_AVAILABLE = False
    st.error("Erro: A chave GEMINI_API_KEY não foi encontrada no Streamlit Secrets. A análise com IA não funcionará.")
except Exception as e:
    GEMINI_AVAILABLE = False
    st.error(f"Erro ao configurar a API Gemini: {e}")


# === FUNÇÕES AUXILIARES ===

def extract_text_from_pdf(uploaded_file):
    """Extrai texto de um PDF carregado usando PyMuPDF (fitz)."""
    if not PDF_LIB_AVAILABLE:
        return ""
        
    text = ""
    try:
        # Lógica para PyMuPDF (fitz): Abre o arquivo em memória (stream)
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        # Reposiciona o ponteiro do arquivo para que o Streamlit possa processá-lo novamente
        uploaded_file.seek(0)
        return text.strip()
    except Exception as e:
        st.error(f"Erro fatal ao extrair texto do PDF com PyMuPDF: {e}")
        return ""

def analyze_with_gemini(text):
    """Usa o Gemini para detectar, quantificar e corrigir vieses."""
    if not GEMINI_AVAILABLE:
        return None

    # O exemplo agora usa descrições em vez de valores fixos (como 45)
    prompt = f"""
Você é um assistente jurídico especializado em análise de vieses.
Sua tarefa é analisar o texto abaixo (trechos de decisões judiciais) e:
1. Identificar todos os tipos de viés presentes (de gênero, racial, socioeconômico, moral e cognitivo).
2. Estimar uma porcentagem geral de viés no texto (de 0 a 100).
3. Explicar brevemente cada ocorrência.
4. Reescrever o trecho problemático de forma neutra, sem alterar o conteúdo jurídico.
5. Gerar um relatório textual resumido dos achados.
6. Gerar o texto completo da decisão reescrito de forma neutra, usando as sugestões.
7. IMPORTANTE: Assegure-se de que o JSON seja 100% válido. Escape corretamente todas as aspas (") e quebras de linha (\\n) dentro dos campos de texto do JSON.

Texto:
{text}

Responda **exclusivamente** no formato JSON, usando este modelo como guia:
{{
  "porcentagem_vies": <numero_inteiro_calculado_de_0_a_100>,
  "relatorio_resumo": "<relatorio_textual_resumido_dos_achados_da_analise>",
  "analise": [
    {{
      "tipo": "<tipo_de_vies_encontrado_ex: 'gênero'>",
      "trecho": "<trecho_original_exato_com_vies>",
      "explicacao": "<explicacao_curta_do_porque_ha_vies>",
      "sugestao": "<reescrita_neutra_do_trecho>"
    }},
    {{
      "tipo": "<tipo_de_vies_encontrado_ex: 'moral, cognitivo'>",
      "trecho": "<outro_trecho_original_com_vies>",
      "explicacao": "<explicacao_curta_do_porque_ha_vies>",
      "sugestao": "<reescrita_neutra_do_trecho>"
    }}
  ],
  "texto_reescrito": "<texto_completo_reescrito_de_forma_neutra_e_juridicamente_valida>"
}}
"""

    try:
        # Usando o nome estável 'gemini-2.5-flash'
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text
    except Exception as e:
        st.error(f"Erro na chamada da API Gemini: {e}")
        return None

# Parser de JSON robusto para lidar com respostas malformadas
def parse_and_display_results(json_str):
    """Analisa o JSON e exibe os resultados na interface Streamlit. (Função Robusta)"""
    
    data = None
    
    try:
        # Tentativa 1: Analisar o JSON diretamente
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        st.warning(f"Erro ao analisar JSON (Detalhe: {e}). Tentando limpar e extrair o JSON do texto...")
        
        # Tentativa 2: Extrair o JSON do texto (caso a IA adicione "Aqui está...")
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            cleaned_json_str = match.group(0)
            try:
                # Tentativa 3: Analisar o JSON limpo
                data = json.loads(cleaned_json_str)
            except json.JSONDecodeError as e2:
                st.error(f"Erro final ao analisar JSON. A IA gerou um JSON corrompido (provavelmente aspas não escapadas). Detalhe: {e2}")
                st.code(json_str) 
                return None
        else:
            st.error("Erro: Nenhum JSON foi encontrado na resposta da IA.")
            st.code(json_str)
            return None
    
    # Se 'data' foi carregado com sucesso (em qualquer tentativa), continua a exibir
    try:
        
        col_perc, col_tematica = st.columns([1, 3])
        
        porcentagem = data.get('porcentagem_vies', 'N/A')
        col_perc.metric("Viés Estimado", f"**{porcentagem}%**")
        col_tematica.markdown(f"#### Classificação Temática: **{st.session_state.get('classificacao_tematica', 'Não Classificado')}**")
        
        # === 1. GRÁFICO DE DISTRIBUIÇÃO DE VIÉS (ALTAIR) ===
        analysis_list = data.get('analise', [])
        if analysis_list:
            
            all_biases_list = []
            for item in analysis_list:
                types = item.get('tipo', 'desconhecido').split(',')
                all_biases_list.extend([t.strip() for t in types])
            
            bias_counts = pd.Series(all_biases_list).value_counts().reset_index()
            bias_counts.columns = ['Tipo de Viés', 'Contagem']

            st.markdown("---")
            st.markdown("### 📈 Distribuição dos Tipos de Viés")
            
            chart = alt.Chart(bias_counts).mark_bar().encode(
                x=alt.X('Tipo de Viés:N', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Contagem:Q', title='Ocorrências'),
                color=alt.Color('Tipo de Viés:N', legend=None),
                tooltip=['Tipo de Viés', 'Contagem']
            ).properties(
                title='Contagem de Ocorrências por Tipo de Viés'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            st.markdown("---")
        
        # === 2. RELATÓRIO E REESCRITA ===
        st.markdown("### 📋 Relatório Resumido")
        st.info(data.get('relatorio_resumo', 'Relatório não disponível.'))
        
        
        # === 3. ANÁLISE DETALHADA ===
        st.markdown("### 🔍 Detalhamento dos Vieses Encontrados")

        if not analysis_list:
             st.info("Nenhum viés significativo detectado na análise.")
        else:
            for item in analysis_list:
                st.markdown(f"#### Tipo de Viés: **{item['tipo'].capitalize()}**")
                
                st.error("🚫 **Trecho Problemático (Viés)**")
                st.code(item.get('trecho', 'N/A'), language='plaintext')
                
                st.success("✅ **Sugestão de Reescrita (Neutro)**")
                st.code(item.get('sugestao', 'N/A'), language='plaintext')
                
                st.warning("💡 **Explicação:**")
                st.write(item.get('explicacao', 'N/A'))
                st.markdown("---")


        # === 4. TEXTO REESCRITO (AGORA POR ÚLTIMO E OPCIONAL) ===
        st.markdown("### ✍️ Texto Completo Reescrito")
        
        with st.expander("Clique aqui para gerar o texto completo reescrito (Neutro)"):
            st.code(data.get('texto_reescrito', 'Reescrita não disponível.'), language='plaintext')
        
        return data
        
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao exibir os resultados: {e}")
        return None

# ==================================================================
# === REMOVIDO ===
# A função inteira 'save_to_gsheets' foi removida.
# ==================================================================


# === INTERFACE PRINCIPAL DO APP ===

st.title("⚖️ DetectaBias – Analisador de Viés Jurídico")
st.write("Ferramenta web voltada à identificação de viés linguístico e argumentativo em decisões judiciais.")

# Inicializa o estado
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None
if 'original_text' not in st.session_state:
    st.session_state.original_text = None
if 'classificacao_tematica' not in st.session_state:
    st.session_state.classificacao_tematica = 'Não Classificado'
# === CORREÇÃO PDF: Inicializa o estado de impressão ===
if 'printing' not in st.session_state:
    st.session_state.printing = False
# ===================================================

with st.sidebar:
    st.header("⚙️ Configurações de Análise")
    st.session_state.classificacao_tematica = st.selectbox(
        "Selecione a Classificação Temática (Opcional)",
        ["Não Classificado", "Direito Penal", "Família", "Trabalhista", "Cível", "Administrativo", "Outro"]
    )
    
    uploaded_file = st.file_uploader("📄 Envie o arquivo PDF da decisão", type=["pdf"])

if uploaded_file:
    with st.spinner("Extraindo texto do PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.original_text = text

    if st.session_state.original_text:
        st.success("Texto extraído com sucesso! ✅")

        with st.expander("📘 Texto da Decisão (Original)"):
            st.text_area("Texto extraído:", st.session_state.original_text, height=300)

        if st.button("🔍 Analisar com IA", type="primary"):
            st.session_state.parsed_data = None
            
            with st.spinner("Analisando vieses com o Gemini..."):
                analysis_json_str = analyze_with_gemini(st.session_state.original_text)

            if analysis_json_str:
                st.session_state.parsed_data = parse_and_display_results(analysis_json_str)

        # === REMOVIDO ===
        # Bloco do Google Sheets removido.
        # =================

        # === BLOCO DE DOWNLOAD DE PDF (LÓGICA CORRIGIDA) ===
        if st.session_state.parsed_data:
            
            # 1. Cria a seção de download
            st.markdown('<div class="download-pdf-section">', unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("📁 Salvar Relatório da Análise")
            st.markdown("<p>Clique no botão abaixo para abrir a caixa de diálogo de impressão e salvar o relatório como PDF.</p>", unsafe_allow_html=True)

            if st.button("💾 Salvar Análise em PDF", key="download_pdf"):
                # 2. Seta a flag de impressão para True
                st.session_state.printing = True
            
            st.markdown('</div>', unsafe_allow_html=True) 

            # 3. Bloco de execução (executado no re-run após clicar no botão)
            if st.session_state.printing:
                # 3a. Injeta o CSS de impressão
                print_css = """
                <style>
                @media print {
                    /* Esconde a barra lateral */
                    [data-testid="stSidebar"] {
                        display: none;
                    }
                    /* Esconde o cabeçalho (Deploy, etc) */
                    [data-testid="stHeader"] {
                        display: none;
                    }
                    /* Esconde o botão de "Analisar" (botão primário) */
                    button[kind="primary"] {
                        display: none;
                    }
                    /* Esconde o expander do texto original */
                    [data-testid="stExpander"][label="📘 Texto da Decisão (Original)"] {
                        display: none;
                    }
                    /* Esconde a seção de download (o próprio botão/texto) */
                    .download-pdf-section {
                        display: none;
                    }
                    /* Garante que os st.code (trechos) não sejam cortados */
                    .stCode {
                        white-space: pre-wrap !important;
                        word-wrap: break-word !important;
                    }
                }
                </style>
                """
                st.markdown(print_css, unsafe_allow_html=True)
                
                # 3b. Executa o JavaScript com um atraso de 1.5s
                # O atraso é crucial para o CSS ser aplicado antes da impressão
                components.html(
                    "<script>setTimeout(window.print, 1500);</script>",
                    height=0,
                )
                
                # 3c. Reseta a flag
                st.session_state.printing = False
        # =================================

    else:
        st.warning("O texto extraído do PDF está vazio.")
else:
    st.info("Envie um arquivo PDF para começar a análise.")