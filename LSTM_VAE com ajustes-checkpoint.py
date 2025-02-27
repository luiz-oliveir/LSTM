import numpy as np
np.random.seed(0)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
tf.random.set_seed(0)
import glob
from keras_tuner import RandomSearch
import sys

# Configuração otimizada para GPU NVIDIA/CUDA
def setup_gpu():
    try:
        # Listar GPUs disponíveis
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("GPUs disponíveis:", len(gpus))
            for gpu in gpus:
                print(" -", gpu.name)
            
            # Permitir crescimento de memória dinâmico
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Configurar para formato de dados mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print("\nGPU configurada com sucesso!")
            print("Usando mixed precision:", policy.name)
            
            # Verificar se CUDA está disponível
            if tf.test.is_built_with_cuda():
                print("CUDA está disponível")
                print("Versão do CUDA:", tf.sysconfig.get_build_info()["cuda_version"])
            
            # Configurar cache de kernels CUDA
            tf.config.experimental.enable_op_determinism()
            
            return True
        else:
            print("Nenhuma GPU encontrada. Usando CPU.")
            return False
    except Exception as e:
        print(f"Erro ao configurar GPU: {e}")
        print("Continuando com CPU")
        return False

# Configurar GPU no início do script
using_gpu = setup_gpu()

from tensorflow import keras, data
import tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers, activations, optimizers
from tensorflow.keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt

dataset_name = "bearing_dataset"  # Apenas para referência
#train_ratio = 0.75
row_mark = 740
batch_size = 128
time_step = 1
x_dim = 4
lstm_h_dim = 8
z_dim = 4
epoch_num = 100
threshold = None

mode = 'train'
model_dir = "./lstm_vae_model/"
image_dir = "./lstm_vae_images/"
data_dir = r"C:\F_analises\INMET\Convencionais processadas temperaturas"

# Criar diretórios necessários
os.makedirs(model_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Parâmetros de ativação
lstm_activation = 'softplus'  # Pode mudar para 'tanh', 'relu', etc
sigma_activation = 'tanh'     # Ativação para sigma_x

def split_normalize_data(all_df):
    #row_mark = int(all_df.shape[0] * train_ratio)
    train_df = all_df[:row_mark]
    test_df = all_df[row_mark:]

    scaler = MinMaxScaler()
    scaler.fit(np.array(all_df)[:, 1:])
    train_scaled = scaler.transform(np.array(train_df)[:, 1:])
    test_scaled = scaler.transform(np.array(test_df)[:, 1:])
    return train_scaled, test_scaled

def reshape(da):
    return da.reshape(da.shape[0], time_step, da.shape[1]).astype("float32")

class Sampling(layers.Layer):
    def __init__(self, name='sampling_z'):
        super(Sampling, self).__init__(name=name)
    
    def call(self, inputs):
        mu, logvar = inputs
        print('mu: ', mu)
        sigma = K.exp(logvar * 0.5)
        epsilon = K.random_normal(shape=(mu.shape[0], z_dim), mean=0.0, stddev=1.0)
        return mu + epsilon * sigma
    
    def get_config(self):
        config = super(Sampling, self).get_config()
        config.update({'name': self.name})
        return config

class Encoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='encoder', activation=lstm_activation, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.encoder_inputs = keras.Input(shape=(time_step, x_dim))
        self.encoder_lstm = layers.LSTM(
            lstm_h_dim, 
            activation=activation,  # Usar parâmetro
            name='encoder_lstm', 
            stateful=True
        )
        self.z_mean = layers.Dense(z_dim, name='z_mean')
        self.z_logvar = layers.Dense(z_dim, name='z_log_var')
        self.z_sample = Sampling()
    
    def call(self, inputs):
        self.encoder_inputs = inputs
        hidden = self.encoder_lstm(self.encoder_inputs)
        mu_z = self.z_mean(hidden)
        logvar_z = self.z_logvar(hidden)
        z = self.z_sample((mu_z, logvar_z))
        return mu_z, logvar_z, z
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'name': self.name,
            'z_sample': self.z_sample.get_config()
        })
        return config

class Decoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='decoder', activation=lstm_activation, sigma_activation=sigma_activation, **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.z_inputs = layers.RepeatVector(time_step, name='repeat_vector')
        self.decoder_lstm_hidden = layers.LSTM(
            lstm_h_dim, 
            activation=activation,
            return_sequences=True, 
            name='decoder_lstm'
        )
        self.x_mean = layers.Dense(x_dim, name='x_mean')
        self.x_sigma = layers.Dense(
            x_dim, 
            name='x_sigma', 
            activation=sigma_activation  # Usar parâmetro
        )
    
    def call(self, inputs):
        z = self.z_inputs(inputs)
        hidden = self.decoder_lstm_hidden(z)
        mu_x = self.x_mean(hidden)
        sigma_x = self.x_sigma(hidden)
        return mu_x, sigma_x
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'name': self.name
        })
        return config

loss_metric = keras.metrics.Mean(name='loss')
likelihood_metric = keras.metrics.Mean(name='log likelihood')

class LSTM_VAE(keras.Model):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='lstm_vae', **kwargs):
        super(LSTM_VAE, self).__init__(name=name, **kwargs)

        self.encoder = Encoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)
        self.decoder = Decoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)
    
    def call(self, inputs):
        mu_z, logvar_z, z = self.encoder(inputs)
        mu_x, sigma_x = self.decoder(z)

        var_z = K.exp(logvar_z)
        kl_loss = K.mean(-0.5 * K.sum(var_z - logvar_z + tf.square(1 - mu_z), axis=1), axis=0)
        self.add_loss(kl_loss)

        dist = tfp.distributions.Normal(loc=mu_x, scale=tf.abs(sigma_x))
        log_px = -dist.log_prob(inputs)

        return mu_x, sigma_x, log_px
    
    def get_config(self):
        config = {
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'name': self.name
        }
        return config
    
    def reconstruct_loss(self, x, mu_x, sigma_x):
        var_x = K.square(sigma_x)
        reconst_loss = -0.5 * K.sum(K.log(var_x), axis=2) + K.sum(K.square(x - mu_x) / var_x, axis=2)
        reconst_loss = K.reshape(reconst_loss, shape=(x.shape[0], 1))
        return K.mean(reconst_loss, axis=0)

    def mean_log_likelihood(self, log_px):
        log_px = K.reshape(log_px, shape=(log_px.shape[0], log_px.shape[2]))
        mean_log_px = K.mean(log_px, axis=1)
        return K.mean(mean_log_px, axis=0)

    def train_step(self, data):
        with tf.device('/GPU:0' if using_gpu else '/CPU:0'):
            with tf.GradientTape() as tape:
                # Usar tf.data para otimizar o pipeline de dados
                if isinstance(data, tf.data.Dataset):
                    x = next(iter(data))
                else:
                    x = data
                
                # Forward pass com mixed precision
                if using_gpu:
                    x = tf.cast(x, tf.float16)
                
                # Forward pass
                mu_z, logvar_z = self.encoder(x)
                z = self.encoder.z_sample([mu_z, logvar_z])
                mu_x, sigma_x = self.decoder(z)
                
                # Calcular loss
                log_px = self.reconstruct_loss(x, mu_x, sigma_x)
                kl_loss = -0.5 * tf.reduce_mean(1 + logvar_z - tf.square(mu_z) - tf.exp(logvar_z))
                loss = -log_px + kl_loss
                
                # Escalar loss para mixed precision
                if using_gpu:
                    loss = tf.cast(loss, tf.float32)
                
                # Atualizar métricas
                loss_metric.update_state(loss)
                likelihood_metric.update_state(log_px)
            
            # Backpropagation otimizado
            variables = self.trainable_variables
            gradients = tape.gradient(loss, variables)
            # Clipar gradientes para evitar explosão
            gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
            self.optimizer.apply_gradients(zip(gradients, variables))
            
            return {
                "loss": loss_metric.result(),
                "log_px": likelihood_metric.result(),
                "kl_loss": kl_loss
            }

def prepare_training_data(all_df, batch_size=128):  # Aumentado batch_size para GPU
    # Preparar dados usando tf.data para pipeline otimizado
    dataset = tf.data.Dataset.from_tensor_slices(all_df.values.astype('float32'))
    
    # Configurar pipeline otimizado
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=min(len(all_df), 10000))  # Buffer size limitado para memória
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    if using_gpu:
        # Usar estratégia de distribuição para GPU
        strategy = tf.distribute.MirroredStrategy()
        dataset = strategy.experimental_distribute_dataset(dataset)
    
    return dataset

def generate_sample_data():
    """Generate sample data if no Excel files are found."""
    print("\nGenerating sample data for testing...")
    
    # Create sample timestamps
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    n_samples = len(dates)
    
    # Generate synthetic data
    np.random.seed(42)
    data = {
        'timestamp': dates,
        'nivel_agua': np.random.normal(100, 10, n_samples),  # Normal distribution
        'vazao': np.abs(np.random.normal(50, 5, n_samples)),  # Positive values
        'temperatura': np.random.normal(25, 3, n_samples),  # Normal distribution
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some anomalies
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[anomaly_idx, 'nivel_agua'] *= 1.5
    df.loc[anomaly_idx, 'vazao'] *= 2
    df.loc[anomaly_idx, 'temperatura'] += 10
    
    # Save to Excel file
    sample_file = os.path.join(data_dir, 'sample_data.xlsx')
    print(f"Saving sample data to {sample_file}")
    df.to_excel(sample_file, index=False)
    return df

def load_and_prepare_data():
    print("Loading and preparing data...")
    
    # Diretório onde estão os arquivos Excel
    dfs = []
    
    # Get list of Excel files (ignorando arquivos temporários que começam com ~$)
    excel_files = [f for f in glob.glob(os.path.join(data_dir, "*.xlsx")) if not os.path.basename(f).startswith('~$')]
    
    if not excel_files:
        print(f"\nWARNING: No Excel files found in {data_dir}")
        print("Please make sure the Excel files are in the correct directory.")
        sys.exit(1)
    else:
        # Load and combine data from all files
        for file in excel_files:
            print(f"Loading {os.path.basename(file)}...")
            try:
                df = pd.read_excel(file)
                
                # Verificar se as colunas necessárias existem
                required_cols = ['Dia', 'Mes', 'Ano']
                missing = [col for col in required_cols if col not in df.columns]
                
                # Procurar a coluna de temperatura (pode ter variações no nome)
                temp_col = None
                for col in df.columns:
                    if 'TEMPERATURA MAXIMA' in str(col).upper():
                        temp_col = col
                        break
                
                if temp_col is None:
                    print(f"Aviso: {os.path.basename(file)} não possui coluna de temperatura máxima")
                    continue
                
                if missing:
                    print(f"Aviso: {os.path.basename(file)} está faltando colunas {missing}")
                    continue
                
                # Criar coluna de data
                df['Data'] = pd.to_datetime({
                    'year': df['Ano'],
                    'month': df['Mes'],
                    'day': df['Dia']
                })
                
                # Adicionar identificador da estação
                station_id = os.path.splitext(os.path.basename(file))[0]
                df['station_id'] = station_id
                
                # Renomear coluna de temperatura para um nome mais simples
                df = df.rename(columns={temp_col: 'temperatura'})
                
                # Remover linhas com valores nulos na temperatura
                df = df.dropna(subset=['temperatura'])
                
                dfs.append(df)
                print(f"Processado com sucesso: {len(df)} registros")
                
            except Exception as e:
                print(f"Erro ao ler {file}: {e}")
                continue
    
    if not dfs:
        raise ValueError("Nenhum arquivo válido para processar")
    
    # Combine all dataframes
    print("Combining data from all stations...")
    all_df = pd.concat(dfs, ignore_index=True)
    
    # Extrair características temporais
    all_df['year'] = all_df['Data'].dt.year
    all_df['month'] = all_df['Data'].dt.month
    all_df['day'] = all_df['Data'].dt.day
    all_df['dayofweek'] = all_df['Data'].dt.dayofweek
    all_df['is_weekend'] = all_df['Data'].dt.dayofweek >= 5
    
    # Selecionar apenas as colunas numéricas relevantes para o modelo
    feature_cols = ['temperatura', 'year', 'month', 'day', 'dayofweek', 'is_weekend']
    all_df = all_df[feature_cols]
    
    # Update x_dim based on actual number of features
    global x_dim
    x_dim = len(feature_cols)
    print(f"Number of features (x_dim): {x_dim}")
    print(f"Total number of records: {len(all_df)}")
    
    return all_df

def hyperparameter_tuning(train_X):
    def build_model(hp):
        # Parâmetros ajustáveis
        hp_lstm_dim = hp.Int('lstm_dim', min_value=4, max_value=32, step=4)
        hp_z_dim = hp.Int('z_dim', min_value=2, max_value=16, step=2)
        hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        
        model = LSTM_VAE(
            time_step, 
            train_X.shape[2], 
            hp_lstm_dim, 
            hp_z_dim
        )
        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=hp_learning_rate, 
                amsgrad=True
            )
        )
        return model
    
    tuner = RandomSearch(
        build_model,
        objective='loss',
        max_trials=10,
        executions_per_trial=2,
        directory='tuner_results',
        project_name='lstm_vae_tuning'
    )
    
    print("\nIniciando busca de hiperparâmetros...")
    tuner.search(train_X, epochs=50, batch_size=batch_size)
    
    print("\nMelhores hiperparâmetros encontrados:")
    print(tuner.get_best_hyperparameters(num_trials=1)[0].values)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

def plot_loss_moment(history):
    _, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'blue', label='Loss', linewidth=1)
    ax.plot(history['log_likelihood'], 'red', label='Log likelihood', linewidth=1)
    ax.set_title('Loss and log likelihood over epochs')
    ax.set_ylabel('Loss and log likelihood')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(image_dir + 'loss_lstm_vae_' + mode + '.png')

def plot_log_likelihood(df_log_px):
    plt.figure(figsize=(14, 6), dpi=80)
    plt.title("Log likelihood")
    sns.set_color_codes()
    sns.distplot(df_log_px, bins=40, kde=True, rug=True, color='blue')
    plt.savefig(image_dir + 'log_likelihood_' + mode + '.png')

def save_model(model):
    with open(model_dir + 'lstm_vae.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(model_dir + 'lstm_vae_ckpt')

def load_model():
    lstm_vae_obj = {'Encoder': Encoder, 'Decoder': Decoder, 'Sampling': Sampling}
    with keras.utils.custom_object_scope(lstm_vae_obj):
        with open(model_dir + 'lstm_vae.json', 'r'):
            model = keras.models.model_from_json(model_dir + 'lstm_vae.json')
        model.load_weights(model_dir + 'lstem_vae_ckpt')
    return model

def main():
    print("\nIniciando processamento...")
    if using_gpu:
        print("Executando com otimização de GPU")
    else:
        print("Executando com configuração padrão")
        
    # Data loading
    print("Loading and preparing data...")
    all_df = load_and_prepare_data()
    
    # Pre-processing
    print("Pre-processing data...")
    train_scaled, test_scaled = split_normalize_data(all_df)
    print("train and test data shape after scaling: ", train_scaled.shape, test_scaled.shape)

    train_X = reshape(train_scaled)
    test_X = reshape(test_scaled)
    
    if mode == "train":
        print("\nIniciando otimização de hiperparâmetros...")
        model = hyperparameter_tuning(train_X)
        
        # Treinar com os melhores hiperparâmetros
        history = model.fit(
            train_X, 
            epochs=epoch_num, 
            batch_size=batch_size, 
            shuffle=False
        ).history
        
        model.summary()
        
        # Calcular threshold dinâmico (95º percentil dos erros de treino)
        _, _, train_log_px = model.predict(train_X)
        train_errors = np.mean(train_log_px, axis=(1,2))
        global threshold
        threshold = np.percentile(train_errors, 95)  # 95% dos dados normais
        
        # Salvar threshold junto com o modelo
        np.save(os.path.join(model_dir, 'threshold.npy'), threshold)
        print(f"Auto Threshold: {threshold:.4f}")
        
        plot_loss_moment(history)
        save_model(model)
        
    elif mode == "infer":
        model = load_model()
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001, epsilon=1e-6, amsgrad=True))
        
        # Carregar threshold salvo
        threshold = np.load(os.path.join(model_dir, 'threshold.npy'))
    else:
        print("Unknown mode: ", mode)
        exit(1)
    
    _, _, train_log_px = model.predict(train_X, batch_size=1)
    train_log_px = train_log_px.reshape(train_log_px.shape[0], train_log_px.shape[2])
    df_train_log_px = pd.DataFrame()
    df_train_log_px['log_px'] = np.mean(train_log_px, axis=1)
    plot_log_likelihood(df_train_log_px)

    
    _, _, test_log_px = model.predict(test_X, batch_size=1)
    test_log_px = test_log_px.reshape(test_log_px.shape[0], test_log_px.shape[2])
    df_log_px = pd.DataFrame()
    df_log_px['log_px'] = np.mean(test_log_px, axis=1)
    df_log_px = pd.concat([df_train_log_px, df_log_px])
    df_log_px['threshold'] = threshold
    df_log_px['anomaly'] = df_log_px['log_px'] > df_log_px['threshold']
    df_log_px.index = np.array(all_df)[:, 0]
    
    df_log_px.plot(logy=True, figsize=(16, 9), color=['blue', 'red'])
    plt.savefig(image_dir + 'anomaly_lstm_vae_' + mode + '.png')
    
if __name__ == "__main__":
    main()