import json
import subprocess
import time
import os
import sys

# Diretório raiz do projeto
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAST_CLI = os.path.join(ROOT_DIR, "vast")

# Carregar .env manualmente se existir para obter a VAST_API_KEY
env_path = os.path.join(ROOT_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip()

def run_vast(args, raw=False):
    """Executa um comando do vast CLI e retorna o stdout."""
    api_key = os.environ.get("VAST_API_KEY")
    cmd = [VAST_CLI]
    if api_key:
        cmd += ["--api-key", api_key]
    cmd += args
    if raw:
        cmd.append("--raw")
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Mascarar a chave de API na saída de erro por motivos de segurança
        masked_cmd = " ".join([arg if arg != api_key else "********" for arg in cmd])
        print(f"[ERRO VAST] Comando {masked_cmd} falhou:")
        print(res.stderr)
        raise RuntimeError(res.stderr)
    return res.stdout.strip()

def main():
    print("="*60)
    print("      INICIANDO PIPELINE AUTOMÁTICO DE TREINAMENTO NO VAST.AI      ")
    print("="*60)
    
    # 1. Verificar saldo do usuário
    user_info = json.loads(run_vast(["show", "user"], raw=True))
    balance = user_info.get("credit", 0.0)
    print(f"[INFO] Saldo disponível no Vast.ai: ${balance:.2f}")
    
    if balance < 1.0:
        print("[ERRO] Saldo muito baixo para iniciar o treinamento. Adicione fundos no Vast.ai.")
        sys.exit(1)
        
    # 2. Buscar a oferta mais barata verificada de RTX 4090 em Datacenter confiável (confiabilidade > 99%, velocidade > 200Mbps e fora da China)
    print("[INFO] Buscando a GPU RTX 4090 em Datacenter, verificada, rápida e confiável...")
    offers = json.loads(run_vast(["search", "offers", "gpu_name=RTX_4090 verified=true datacenter=true direct_port_count>0 reliability>0.99 inet_down>200 geolocation!=CN"], raw=True))
    
    if not offers:
        print("[ERRO] Nenhuma GPU RTX 4090 verificada disponível no momento.")
        sys.exit(1)
        
    # Ordenar por dph_total (dollar per hour)
    offers = sorted(offers, key=lambda x: x.get("dph_total", 999.0))
    cheapest = offers[0]
    
    offer_id = cheapest["id"]
    dph = cheapest.get("dph_total", 0.0)
    dl_speed = cheapest.get("inet_down", 0.0)
    ul_speed = cheapest.get("inet_up", 0.0)
    reliability = cheapest.get("reliability2", 0.0)
    
    print(f"\n[SUCESSO] Melhor GPU encontrada:")
    print(f"  * ID da Oferta: {offer_id}")
    print(f"  * Custo: ${dph:.4f}/hora")
    print(f"  * Conexão: DL {dl_speed:.1f} Mbps / UL {ul_speed:.1f} Mbps")
    print(f"  * Confiabilidade: {reliability * 100:.2f}%")
    
    if dph > 0.40:
        print(f"[AVISO] O preço de ${dph:.2f}/hora está alto. O orçamento de $4.15 pode ser consumido se esquecido.")
        if os.environ.get("FORCE_YES") == "1" or not sys.stdin.isatty():
            print("Continuando automaticamente (FORCE_YES=1 ou stdin não-interativo)...")
            confirm = 's'
        else:
            confirm = input("Deseja continuar mesmo assim? (s/n): ").strip().lower()
        if confirm != 's':
            print("[CANCELADO] Operação abortada pelo usuário.")
            sys.exit(0)
            
    # 3. Alugar a instância
    print(f"\n[INFO] Alugando a instância {offer_id} (Disk: 32GB)...")
    # Imagem oficial PyTorch compatível com CUDA 12.1
    image = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"
    rent_res = run_vast(["create", "instance", str(offer_id), "--image", image, "--disk", "32"], raw=True)
    rent_data = json.loads(rent_res)
    
    instance_id = rent_data.get("new_contract")
    if not instance_id:
        print("[ERRO] Falha ao alugar a instância. Detalhes:")
        print(rent_res)
        sys.exit(1)
        
    print(f"[SUCESSO] Instância alugada com Contrato ID: {instance_id}")
    
    instance_running = False
    ssh_host = None
    ssh_port = None
    
    try:
        # 4. Aguardar o boot da instância
        print("[INFO] Aguardando inicialização da máquina (isso pode levar de 1 a 3 minutos)...")
        while not instance_running:
            time.sleep(10)
            instances = json.loads(run_vast(["show", "instances"], raw=True))
            my_instance = next((inst for inst in instances if inst.get("id") == instance_id), None)
            
            if not my_instance:
                print(f"[ERRO] A instância {instance_id} não foi encontrada na sua conta.")
                break
                
            status = my_instance.get("status_msg", "unknown")
            print(f"  * Status atual: '{status}'")
            
            if my_instance.get("actual_status") == "running" and my_instance.get("ssh_port") is not None:
                ssh_host = my_instance.get("ssh_host")
                ssh_port = my_instance.get("ssh_port")
                instance_running = True
                print(f"\n[SUCESSO] Máquina online!")
                print(f"  * Host: {ssh_host}")
                print(f"  * Porta SSH: {ssh_port}")
                
        if not instance_running:
            raise RuntimeError("Falha ao inicializar a instância.")
            
        # Pequena pausa para garantir que o daemon SSH da máquina virtual esteja pronto para receber conexões
        print("[INFO] Aguardando 30 segundos para estabilização do SSH e injeção de chaves...")
        time.sleep(30)
        
        # 5. Fazer upload dos arquivos necessários via SCP (com retentativas)
        print("\n[INFO] Enviando arquivos de código para a instância remota...")
        # Definir parâmetros de conexão SSH segura e silenciosa
        ssh_opts = f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_ed25519 -P {ssh_port}"
        
        # Enviar pasta src e os scripts principais
        # Vamos compactar localmente para fazer um único upload rápido
        print("  * Compactando código fonte localmente...")
        subprocess.run(f"tar -czf temp_code.tar.gz src/ generate_synthetic_conversations.py train_hybrid_1b.py run_multiturn_tests.py", shell=True, cwd=ROOT_DIR)
        
        print("  * Enviando arquivo tar.gz (com retentativas)...")
        scp_cmd = f"scp {ssh_opts} {ROOT_DIR}/temp_code.tar.gz root@{ssh_host}:/root/temp_code.tar.gz"
        
        max_retries = 10
        scp_success = False
        for attempt in range(max_retries):
            res = subprocess.run(scp_cmd, shell=True)
            if res.returncode == 0:
                scp_success = True
                print("  * Upload concluído com sucesso!")
                break
            else:
                print(f"  * Tentativa {attempt + 1}/{max_retries} falhou (SSH iniciando). Aguardando 15 segundos para reinjeção da chave pública...")
                time.sleep(15)
                
        if not scp_success:
            raise RuntimeError("Falha persistente de conexão SSH/SCP com a instância.")
            
        # Deletar arquivo tar.gz local
        os.remove(os.path.join(ROOT_DIR, "temp_code.tar.gz"))
        
        # 6. Executar comandos remotos de extração, instalação e treinamento
        print("\n[INFO] Iniciando preparação e treinamento remoto...")
        remote_setup_cmd = (
            "mkdir -p /root/project && tar -xzf /root/temp_code.tar.gz -C /root/project && "
            "echo 'torch==2.2.0' > /tmp/constraints.txt && "
            "python3 -m pip install --default-timeout=1000 -c /tmp/constraints.txt 'transformers<5.0.0' 'peft<1.0' bitsandbytes accelerate 'trl<1.0' safetensors hf-transfer && "
            "python3 -c 'import transformers, peft, bitsandbytes' && "
            "cd /root/project && "
            "python3 -u generate_synthetic_conversations.py && "
            "HF_HUB_ENABLE_HF_TRANSFER=1 python3 -u train_hybrid_1b.py --model_id Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --switch_epoch 1 --batch_size 1 --lr 2e-5 --out_dir checkpoints/think_vetor_1b_hybrid_lora && "
            "python3 -u run_multiturn_tests.py && "
            "tar -czf /root/think_vetor_1b_hybrid_lora.tar.gz checkpoints/think_vetor_1b_hybrid_lora/"
        )
        
        ssh_cmd = f"ssh -p {ssh_port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_ed25519 root@{ssh_host} \"{remote_setup_cmd}\""
        print("  * Rodando treinamento (GRPO-RL)... Isso deve demorar de 5 a 10 minutos.")
        print("  * Aguarde... O progresso será exibido abaixo:")
        print("-" * 60)
        
        # Executar mantendo saída em tempo real
        proc = subprocess.Popen(ssh_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        
        if proc.returncode != 0:
            raise RuntimeError("O treinamento remoto falhou.")
            
        print("-" * 60)
        print("[SUCESSO] Treinamento concluído na GPU remota!")
        
        # 7. Baixar o arquivo resultante com os adaptadores LoRA
        print("\n[INFO] Baixando os adaptadores LoRA treinados para a máquina local...")
        os.makedirs(os.path.join(ROOT_DIR, "checkpoints"), exist_ok=True)
        download_cmd = f"scp {ssh_opts} root@{ssh_host}:/root/think_vetor_1b_hybrid_lora.tar.gz {ROOT_DIR}/checkpoints/think_vetor_1b_hybrid_lora.tar.gz"
        subprocess.run(download_cmd, shell=True, check=True)
        
        # Extrair na pasta local
        print("  * Extraindo adaptadores...")
        subprocess.run(f"tar -xzf {ROOT_DIR}/checkpoints/think_vetor_1b_hybrid_lora.tar.gz -C {ROOT_DIR}/", shell=True)
        os.remove(os.path.join(ROOT_DIR, "checkpoints", "think_vetor_1b_hybrid_lora.tar.gz"))
        
        print("\n[FINALIZADO] Adaptadores instalados com sucesso na pasta 'checkpoints/think_vetor_1b_hybrid_lora/'!")

    finally:
        # 8. Garantia ABSOLUTA de destruição da instância para não vazar saldo
        print(f"\n[IMPORTANTE] Limpando recursos... Destruindo instância Vast.ai ID: {instance_id}")
        run_vast(["destroy", "instance", str(instance_id)])
        print("[INFO] Instância terminada. Nenhum custo residual será cobrado.")
        print("="*60)

if __name__ == "__main__":
    main()
