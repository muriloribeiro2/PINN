#Desenvolvido por Murilo Henrique da Silva Ribeiro - 2025
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import load_model
import socket


# Configuração do servidor para comunicação com o código C++
def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 8080))  # Escolha uma porta e endereço apropriados
    server_socket.listen(1)
    print("Aguardando conexão do cliente C++...")
    
    while True:
        conn, addr = server_socket.accept()
        print(f"Conectado a {addr}")
    
        try:
            while True:
                buffer_size = 1024
                file_name = conn.recv(buffer_size)
                file_name = file_name.decode()
                if not file_name:
                    print("Cliente desconectado")
                    break
                print(f"Predizer arquivo {file_name}")
                
                # Chama a função de previsão Fi para cada arquivo recebido (interno e externo)  do código C++
                CalculaFi(file_name, conn)
                print("Processamento concluído para", file_name)
            
        except Exception as e:
            print(f"Ocorreu um erro durante o processamento: {e}")
    
        finally:
            conn.close()
            print("Conexão encerrada com o cliente.")

# Função para salvar os Fis previstos em um arquivo .txt
def save_results_to_txt(file_name, data):
    np.savetxt(file_name, data, fmt='%.6f')

# Carrega os dados do arquivo (Interpolação Externo e Interno) texto para váriaveis especificas de acordo com cada entrada (12 colunas)
def load_data_from_txt(file_path):
    data = np.loadtxt(open(file_path, "rb"), delimiter=' ', usecols=np.arange(0,12))
    #fabsxu, fabsyu, fabszu, u[i][j][k], fabsxv, fabsyv, fabszv, v[i][j][k], fabsxw, fabsyw, fabszw, w[i][j][k],
    return data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7], data[:,8], data[:,9], data[:,10], data[:,11]

# Carrega os dados do arquivo Saida_Interno_1 (ufk2, vfk2 e wfk2)
def carregaU_from_txt(file_path):
    data = np.loadtxt(open(file_path, "rb"), delimiter=' ', usecols=np.arange(0,3))
    #ufk2, vfk2 e wfk2
    return data[:,0], data[:,1], data[:,2]

#Atribuição dos dados dos arquivos para as respectivas variaveis
def AtribuiDataToVectors(data_array, predict=True):
    #se for o arquivo das velocidades
    if(data_array == "Saida_Interno_1.txt"):
        ufk2, vfk2, wfk2 = carregaU_from_txt(data_array)
        return ufk2, vfk2, wfk2
    else:
        fabsxu, fabsyu, fabszu, u, fabsxv, fabsyv, fabszv, v, fabsxw, fabsyw, fabszw, w = load_data_from_txt(data_array)
        #arquivo de interpolação com as distâncias correspondentes separados em bloco de 125 elementos
        if not predict :
            fi1 = np.concatenate([fabsxu, fabsyu, fabszu])
            fi2 = np.concatenate([fabsxv, fabsyv, fabszv])
            fi3 = np.concatenate([fabsxw, fabsyw, fabszw])
            return fi1, fi2, fi3, u
        else:
            fxu = np.reshape(fabsxu, (int(len(fabsxu)/125),125))
            fyu = np.reshape(fabsyu, (int(len(fabsxu)/125),125))
            fzu = np.reshape(fabszu, (int(len(fabsxu)/125),125))
            fxv = np.reshape(fabsxv, (int(len(fabsxv)/125),125))
            fyv = np.reshape(fabsyv, (int(len(fabsxv)/125),125))
            fzv = np.reshape(fabszv, (int(len(fabsxv)/125),125))
            fxw = np.reshape(fabsxw, (int(len(fabsxw)/125),125))
            fyw = np.reshape(fabsyw, (int(len(fabsxw)/125),125))
            fzw = np.reshape(fabszw, (int(len(fabsxw)/125),125))
            u_ = np.reshape(u, (int(len(fabsxw)/125),125))
            v_ = np.reshape(v, (int(len(fabsxw)/125),125))
            w_ = np.reshape(w, (int(len(fabsxw)/125),125))
            
            return fxu, fyu, fzu, fxv, fyv, fzv, fxw, fyw, fzw, u_, v_, w_


#Arquitetura da rede Neural PINN
def create_pinn_model(input_size=125):
    #Função para criar um modelo de rede neural PINN.
    #Parâmetros:
    #input_size (int): Tamanho da entrada, padrão de 125 elementos.
    #Retorna:
    #model (tf.keras.Model): Modelo de rede neural criado.
    model = models.Sequential()  # Inicializa um modelo sequencial
    
    # Camada de entrada com 125 neurônios (correspondendo às 125 posições de entrada)
    model.add(layers.InputLayer(input_shape=(input_size,)))  
    
    # Primeira camada oculta com 512 neurônios e ativação ReLU
    model.add(layers.Dense(512, activation='relu'))
    
    # Segunda camada oculta com 256 neurônios e ativação ReLU
    model.add(layers.Dense(256, activation='relu'))
    
    # Camada de saída com 125 neurônios e ativação softplus
    # Essa ativação garante que os valores previstos sejam positivos
    model.add(layers.Dense(input_size, activation="softplus"))  
    
    return model  # Retorna o modelo criado

#Com base na função WfuncaoF1 (gaussiana) do código C++ e nos dados do resultado da mesma
def compute_fr(fabs):
    
    #Função para calcular o valor de FR(peso) com base no intervalo de Fabs (distâncias).
    #Quanto menor Fabs, maior FR; quanto maior Fabs, menor FR.
    
    # Intervalo 1: Fabs <= 1 (alta influência)
    fr1 = tf.where(fabs <= 1, 
                   0.250000 + (0.499855 - 0.250000) * (1 - fabs), 
                   0.0)
    
    # Intervalo 2: 1 < Fabs <= 2 (influência decrescente)
    fr2 = tf.where((fabs > 1) & (fabs <= 2), 
                   0.000145 + (0.241484 - 0.000145) * (2 - fabs), 
                   0.0)
    
    # Intervalo 3: Fabs > 2 (nenhuma influência)
    fr3 = tf.where(fabs > 2, 0.0, 0.0)
    
    # Combinação dos intervalos
    fr = fr1 + fr2 + fr3
    return fr

#Função de perda para o treinamento do modelo
def loss_function(model, fx, fy, fz, u, ufk):
    #fx, fy, fz = fabsx, fabsy, fabsz para u, v, w, interno e externo, do arquivo texto 
    #u e ufk para u, v, w, interno e externo

    # Converte as variáveis para tensores do TensorFlow com dtype float32
    fx=tf.convert_to_tensor(fx, dtype=tf.float32)
    fy=tf.convert_to_tensor(fy, dtype=tf.float32)
    fz=tf.convert_to_tensor(fz, dtype=tf.float32)
    u=tf.convert_to_tensor(u, dtype=tf.float32)
    ufk=tf.convert_to_tensor(ufk, dtype=tf.float32)

    # Função compute_fr converte os dados reais (distâncias) do arquivo texto em pesos sintéticos para posteriormente comparar com os pesos previstos pelo modelo
    # Predição inicial da rede para FRx, FRy, FRz
    frx = compute_fr(fx)#peso sintetico de X (real) de acordo com a função compute_fr
    pred_frx = model(fx)#peso previsto de X pelo modelo
    fry = compute_fr(fy)#peso sintetico de Y (real) de acordo com a função compute_fr
    pred_fry = model(fy)#peso previsto de Y pelo modelo
    frz = compute_fr(fz)#peso sintetico de Z (real) de acordo com a função compute_fr
    pred_frz = model(fz)#peso previsto de Z pelo modelo

    # Calcula a função de perda Mean Squared Error (MSE)
    # A perda é baseada na diferença quadrática entre os valores previstos (pred_frx, pred_fry, pred_frz) e os valores reais (frx, fry, frz)
    mse_loss = tf.reduce_mean(tf.square(pred_frx - frx) + tf.square(pred_fry - fry) + tf.square(pred_frz - frz))
   
    # Cálculo do produto (fi) de acordo com os pesos previstos pelo modelo
    product = pred_frx*pred_fry*pred_frz
   
       
    # Perda para garantir que a soma dos produtos seja aproximadamente 1
    penalty_loss = tf.square(tf.reduce_sum(product, 1) - 1)
    penalty_loss = tf.reduce_mean(penalty_loss)
    
    #calculo da velocidade interpolada Uk, Vk e Wk do C++
    uk = tf.reduce_sum(product * u, 1)
    
    #cálculo da norma L2 do C++
    square_uk_ufk = tf.square(uk - ufk)
    summation_error = tf.reduce_mean(square_uk_ufk)

    # Perda total
    total_loss = 10*mse_loss + 15*penalty_loss + 10*summation_error
    return total_loss



# Função de treinamento
def train(model, fx, fy, fz, u, ufk, epochs=1000, batch_size=32):
#fx, fy, fz = fabsx, fabsy, fabsz para u, v, w, interno e externo, do arquivo de texto 
#u e ufk para u, v, w, interno e externo do arquito de texto
    
    # Inicializa o otimizador Adam com uma taxa de aprendizado de 0.0001
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = len(fx) // batch_size
        for batch_idx in range(num_batches):
            # Criando lotes de dados
            batch_fx = fx[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_fy = fy[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_fz = fz[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_u = u[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_ufk = ufk[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            with tf.GradientTape() as tape:
                # Calculando a perda para o lote atual
                loss = loss_function(model, batch_fx, batch_fy, batch_fz, batch_u, batch_ufk)
                
            
            # Calcula os gradientes da perda em relação aos pesos do modelo
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Atualiza os pesos do modelo usando o otimizador Adam
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()

        
        # Exibe a perda a cada 100 épocas
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss / num_batches}')
        
    #retorna o modelo treinado
    return model

#função pra realizar previsões utilizando o modelo treinado
def predict(model, fx, fy, fz):
#fx, fy, fz = fabsx, fabsy, fabsz para u, v, w, interno e externo, do arquivo de texto 
   
    # Garantir que as predições são tensores do tipo float32
    pred_frx = tf.constant(fx, dtype=tf.float32)
    pred_fry = tf.constant(fy, dtype=tf.float32)
    pred_frz = tf.constant(fz, dtype=tf.float32)

    #previsão
    pred_frx = model.predict(pred_frx)
    pred_fry = model.predict(pred_fry)
    pred_frz = model.predict(pred_frz)

    #retorna os pesos previstos
    return pred_frx, pred_fry, pred_frz

#Função responsável por realizar o cálculo do fi de acordo com os pesos previstos pelo modelo, salvar o resultado no arquivo texto e enviar para o código C++ por socket
def CalculaFi(arquivo, conn=None):
    #Entrada dos dados do arquivo de texto
    fabsxu, fabsyu, fabszu, fabsxv, fabsyv, fabszv, fabsxw, fabsyw, fabszw, u, v, w = AtribuiDataToVectors(arquivo)
    
    #vetores para armazenar o resultado de fi (1056x125 e 3360x125) tamanho do vetor para cada cilindro
    fiu = np.zeros(len(fabsxu)*125)
    fiv = np.zeros(len(fabsxv)*125)
    fiw = np.zeros(len(fabsxw)*125)

    #carrega os modelos gerados após o treinamento de acordo com cada arquivo e velocidade
    if(arquivo == "Saida_Interpolacao_Interno_1.txt"):
        modelU = load_model('PINNFK_intModelU.h5', compile=False)
        modelV = load_model('PINNFK_intModelV.h5', compile=False)
        modelW = load_model('PINNFK_intModelW.h5', compile=False)
    else:
        modelU = load_model('PINNFK_extModelU.h5', compile=False)
        modelV = load_model('PINNFK_extModelV.h5', compile=False)
        modelW = load_model('PINNFK_extModelW.h5', compile=False)
    contU = 0
    contV = 0
    contW = 0
    
    #pesos previstos pelo modelo para U
    frx, fry, frz = predict(modelU, fabsxu, fabsyu, fabszu)
    for i in range(0, len(fabsxu)):
        for j in range(125):
            #cálculo do fi para U de acordo com os pesos previstos
            fiu[contU] = frx[i][j] * fry[i][j] * frz[i][j]
            contU += 1
       
    #pesos previstos pelo modelo para V
    frx, fry, frz = predict(modelV, fabsxv, fabsyv, fabszv)
    for i in range(0, len(fabsxv)):
        for j in range(125):
            fiv[contV] = frx[i][j] * fry[i][j] * frz[i][j]
            contV += 1
       
    #pesos previstos pelo modelo para W
    frx, fry, frz = predict(modelW, fabsxw, fabsyw, fabszw)
    for i in range(0, len(fabsxw)):
        for j in range(125):
            fiw[contW] = frx[i][j] * fry[i][j] * frz[i][j]
            contW += 1
       
    #salvando em arquivo o resultado da previsão de fi para o cilindro externo e interno
    if arquivo == "Saida_Interpolacao_Externo_1.txt":
        save_results_to_txt("Saida_FiU_Externo.txt", fiu)
        save_results_to_txt("Saida_FiV_Externo.txt", fiv)
        save_results_to_txt("Saida_FiW_Externo.txt", fiw)
    else:
        save_results_to_txt("Saida_FiU_Interno.txt", fiu)
        save_results_to_txt("Saida_FiV_Interno.txt", fiv)
        save_results_to_txt("Saida_FiW_Interno.txt", fiw)
    
    #enviando o resultado da previsão de Fi para o código C++ via conexão cliente/servidor por socket
    data_fiu = fiu.tobytes()
    data_fiv = fiv.tobytes()
    data_fiw = fiw.tobytes()
    if conn is not None:
        conn.sendall(data_fiu)
        conn.sendall(data_fiv)
        conn.sendall(data_fiw)

#função responsável por gerar o modelo de previsão para os cilindros e velocidades após o treinamento
def GeraModelos(externo):
    filenameP = "Saida_Interpolacao_Interno_1.txt"
    filenameV = "Saida_Interno_1.txt"
    nEpochs = 10000 #número de épocas de treinamento para o cilindro interno
    if(externo == 1):
        filenameP = "Saida_Interpolacao_Externo_1.txt"
        nEpochs = 10000 #número de épocas de treinamento para o cilindro externo

    fabsxu, fabsyu, fabszu, fabsxv, fabsyv, fabszv, fabsxw, fabsyw, fabszw, u, v, w = AtribuiDataToVectors(filenameP)
    #Se o modelo a ser pevisto for o externo, os vetores a seguir são zerados, pois não existe velocidade imposta para o cilindro externo
    if(externo == 1):
        ufk2 = np.zeros((len(fabsxu), 1))
        vfk2 = np.zeros((len(fabsxv), 1))
        wfk2 = np.zeros((len(fabsxw), 1))
    else:
        #se o modelo a ser previsto for o interno, os vetores são preenchidos com os dados do arquivo de texto (saida_interno) do código original
        ufk2, vfk2, wfk2 = AtribuiDataToVectors(filenameV) 

    # Inicialização do modelo
    modelU = create_pinn_model()
    modelV = create_pinn_model()
    modelW = create_pinn_model()


    # Compilando o modelo para treinamento
    modelU = train(modelU, fabsxu, fabsyu, fabszu, u, ufk2, epochs=nEpochs)
    if(externo != 1): 
        modelU.save('PINNFK_intModelU.h5')#salvando o modelo para U do cilindro interno após treinado
    else:
        modelU.save('PINNFK_extModelU.h5')#salvando o modelo para U do cilindro externo após treinado
    
    modelV = train(modelV, fabsxv, fabsyv, fabszv, v, vfk2, epochs=nEpochs)
    if(externo != 1):
        modelV.save('PINNFK_intModelV.h5')#salvando o modelo para V do cilindro interno após treinado
    else:
        modelV.save('PINNFK_extModelV.h5')#salvando o modelo para V do cilindro externo após treinado
    
    modelW = train(modelW, fabsxw, fabsyw, fabszw, w, wfk2, epochs=nEpochs)
    if(externo != 1):
        modelW.save('PINNFK_intModelW.h5')#salvando o modelo para W do cilindro interno após treinado
    else:
        modelW.save('PINNFK_extModelW.h5')#salvando o modelo para W do cilindro externo após treinado

#função principal
if __name__ == "__main__":
    #run_server()#servidor - Utilizado somente quando for integrar o código C++ para execução completa do caso
    
    #As 4 funções seguintes podem ser executadas ao mesmo tempo ou de forma independente, desde que o modelo já tenha sido treinado e salvo, somente assim pode-se realizar a previsão de Fi
    GeraModelos(1) #Gera os modelos baseados no cilindro externo
    GeraModelos(0) #Gera os modelos baseados no cilindro interno
    CalculaFi("Saida_Interpolacao_Interno_1.txt")#Realiza a previsão do Fi baseado no modelo do cilindro interno
    CalculaFi("Saida_Interpolacao_Externo_1.txt")#Realiza a previsão do Fi baseado no modelo do cilindro externo