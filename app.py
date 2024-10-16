import os
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

try:
    # Definimos el modelo de llm que vamos a utilizar
    llm = ChatOllama(model="llama3.2:3b")

    if llm is None:
        print("No se pudo cargar el modelo de lenguaje")

    # Definimos el path de los archivos pdf (ruta relativa en este caso)
    pdf_folder_path = "pdfs/"

    # Definimos el directorio donde se va a guardar la base de datos
    persist_db = "chroma_db_dir"

    # Definimos el nombre de la colección
    collection_db = "chroma_collection"

    # Definimos el modelo de embeddings que vamos a utilizar
    embed_model = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(persist_db):

        # Verificamos que solo existan archivos pdfs en el directorio
        if not all([fn.endswith(".pdf") for fn in os.listdir(pdf_folder_path)]):
            raise ValueError("El directorio debe contener solo archivos pdfs")


        # Cargamos los archivos que tengan la extensión .pdf
        loaders = [PyMuPDFLoader(file_path=os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path) if fn.endswith(".pdf")]

        all_pdfs = []

        # Cargamos el contenido del pdf
        
        print("Cargando el contenido de los pdfs...")
        for loader in loaders:
            raw_documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            
            

            # Dividimos el contenido del pdf en chunks
            chunks = text_splitter.split_documents(raw_documents)
            all_pdfs.extend(chunks)

        print("Chunks creados")

        # Creamos la base de datos con los chunks
        print("Creando la base de datos con los chunks...")
        try:
            vs = Chroma.from_documents(
                documents=all_pdfs,
                embedding=embed_model,
                persist_directory=persist_db,
                collection_name=collection_db,
            )
            print("Base de datos creada ", vs)
        except Exception as e:
            print(e)        
        print("Base de datos creada")

    # Creamos el retriever
    print("Creando el retriever...")
    vector_store = Chroma(
        embedding_function=embed_model,
        persist_directory=persist_db,
        collection_name=collection_db,
    )
    print("Retriever creado")

    # Definimos el retriever
    print("Definiendo el retriever...")
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}  # Cantidad de chunks a retornar
    )
    print("Retriever definido")

    # Definimos el template de la pregunta
    print("Definiendo el template de la pregunta...")
    custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
    Si la respuesta no se encuentra en dicha información, di al usuario las preguntas que podes responder. 
    Si la pregunta es muy escueta, puedes pedirle al usuario que la reformule. Si la pregunta es muy amplia, puedes pedirle al usuario
    que la divida en preguntas más pequeñas. Si la pregunta no tiene sentido, puedes pedirle al usuario que la
    reformule. Si la pregunta es ofensiva, puedes pedirle al usuario que se exprese de forma respetuosa.

    Contexto: {context}
    Pregunta: {question}

    Solo devuelve la respuesta útil a continuación y nada más. Responde siempre en español
    Respuesta útil:
    """
    print("Template de la pregunta definido")

    # Definimos el prompt template para la pregunta
    print("Definiendo el prompt template para la pregunta...")
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    print("Prompt template para la pregunta definido")

    # Creamos el chain de QA para realizar la búsqueda
    print("Creando el chain de QA para realizar la búsqueda...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    print("Chain de QA para realizar la búsqueda creado")

    # Realizamos la pregunta al modelo
    quest = input("Ingrese su pregunta: ")
    resp = qa.invoke({"query": quest})

    print(resp["result"])
except Exception as e:
    print(e)
