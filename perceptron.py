import numpy as np

# Variaveis Globais
bias = 1
taxa_de_aprendizado = 0.5

def activation_func(V):
    return 1 if V > 0 else 0
    
def calculate_error(Y_desejado, Y):
    return Y_desejado - Y

def calculate_delta(erro, X_train):
    return taxa_de_aprendizado * erro * X_train

def reconhecer_digito(X_novo, W_train):
    resultado = []
    
    for neuronio in range(2):  # Para cada neurônio (um para "1" e outro para "0")
        v = np.dot(X_novo, W_train[neuronio])  # Produto escalar
        y = activation_func(v)  # Saída do neurônio
        resultado.append(y)
    
    return resultado

def treinar_rede():
        
    # Hiperparâmetros
    num_epocas = 100  # Máximo de épocas para evitar loops infinitos
    limite_erro = 0.001  # Condição de parada

    # Solicita ao usuário o número de amostras
    num_amostras = int(input("Digite a quantidade de amostras de treinamento: "))

    # Inicializando listas para as amostras e os resultados desejados
    X_train = []
    Y_desejado = []

    # Captura as amostras e os resultados desejados
    for i in range(num_amostras):
        print(f"\nAmostra {i+1}:")
        amostra = list(map(int, input("Digite a amostra de treinamento (valores separados por espaço): ").split()))
        digito = int(input("Digite o dígito desejado (0 ou 1): "))
        
        # Adiciona a amostra com o viés (bias)
        X_train.append([bias] + amostra)
        
        # Cria o vetor de resultado desejado com base no dígito
        if digito == 0:
            Y_desejado.append([1, 0])  # Para o número "0"
        elif digito == 1:
            Y_desejado.append([0, 1])  # Para o número "1"
        else:
            print("Dígito inválido! Apenas 0 ou 1 são permitidos.")
            break

    # Converte as listas para numpy arrays
    X_train = np.array(X_train, dtype=np.float64)
    Y_desejado = np.array(Y_desejado, dtype=np.float64)

    # Inicializa as sinapses com valores aleatórios
    W_train = np.random.randn(2, len(X_train[0]))  # 2 neurônios, um para cada dígito
    
    # Treinamento por épocas
    for epoca in range(num_epocas):
        print(f"\n=== Época {epoca+1} ===")
        
        erro_total = 0  # Acumulador de erro para todas as amostras da época
        
        for i in range(len(X_train)):  # Para cada amostra
            print(f"\nTreinando com amostra {i+1} (dígito {Y_desejado[i]})")
            
            for neuronio in range(2):  # Para cada neurônio (um para "1" e outro para "0")
                v = np.dot(X_train[i], W_train[neuronio])  # Produto escalar
                y = activation_func(v)  # Saída do neurônio
                e = calculate_error(Y_desejado[i, neuronio], y)  # Erro
                DeltaW = calculate_delta(e, X_train[i])  # Atualização dos pesos
                
                W_train[neuronio] += DeltaW  # Atualiza os pesos
                
                print(f"\nNeuronio {neuronio}:")
                print(f"  v = {v:.2f}, y = {y}, erro = {e}")
                print(f"  DeltaW: {DeltaW}")
                print(f"  Novos pesos: {W_train[neuronio]}")

                erro_total += e**2  # Acumula o erro quadrático

        # Calcula o erro quadrático médio da época
        erro_medio = erro_total / (2 * len(X_train))  # Normaliza pelo número total de neurônios e amostras
        print(f"\nErro quadrático médio da época {epoca+1}: {erro_medio:.6f}")

        # Condição de parada
        if erro_medio <= limite_erro:
            print("\nCondição de parada atingida! Treinamento encerrado.")
            break

    # Salvando as sinapses (pesos) no arquivo de texto
    np.savetxt('sinapses_final.txt', W_train, fmt='%.2f', delimiter=' ', comments='')

    print("\nSinapses salvas no arquivo 'sinapses_final.txt'.")


def reconhecer_digito_usuario():
    X_novo = []
    # Carregar as sinapses salvas
    W_train_carregado = np.loadtxt('sinapses_final.txt', delimiter=' ')


    amostra = list(map(int, input("\nDigite os valores da amostra para o dígito a ser reconhecido (com espaços): ").split()))

    # Exemplo de nova amostra para teste
    X_novo.append([bias] + amostra)

    # Reconhecimento
    resultado = reconhecer_digito(X_novo, W_train_carregado)
    if resultado[0] == 1 and resultado[1] == 0:
        print("\nDigito reconhecido: Zero")
    elif resultado[0] == 0 and resultado[1] == 1:
        print("\nDigito reconhecido: Um")
    else:
        print("\nDigito não reconhecido")

def main():
    while True:
        print("\nMenu:")
        print("1 - Treinar a rede neural")
        print("2 - Reconhecer um dígito")
        print("3 - Sair")
        
        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            treinar_rede()
        elif opcao == '2':
            reconhecer_digito_usuario()
        elif opcao == '3':
            print("Saindo...")
            break
        else:
            print("Opção inválida! Tente novamente.")

if __name__ == "__main__":
    main()