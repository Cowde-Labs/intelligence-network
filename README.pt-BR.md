# Intelligence Network

[English](README.md) | **Português (Brasil)** | [简体中文](README.zh-CN.md) | [Español](README.es.md)

Execute IA em máquinas independentes sem um servidor central no controle.

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
intelligence up
```

O Intelligence Network é um único binário Rust que transforma uma máquina em um nó. Os nós se encontram por QUIC autenticado, anunciam o que conseguem executar, encaminham trabalhos de inferência e treinamento aos peers adequados, dividem estado entre máquinas e continuam operando quando peers desaparecem. Não é preciso criar uma conta nem acessar uma nuvem, e nada é baixado além do binário. Os modelos só são transferidos se você os colocar na rede.

Este não é um projeto de criptomoedas. Não há token, blockchain, staking ou marketplace. Os nós cooperam porque seus operadores os conectam.

## Início rápido

Linux, macOS e Windows (x86_64 e arm64; no Windows arm64, compilação a partir do código-fonte). O instalador baixa o binário de release e seu checksum do GitHub Releases, verifica o checksum e instala um único arquivo em `~/.local/bin`. Nenhum outro componente é instalado.

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

No Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.ps1 | iex
```

```bash
intelligence up              # cria identidade e configuração e inicia o nó em segundo plano
intelligence status           # saúde, peers, trabalhos e contadores de recursos
intelligence infer "hello"    # encaminha um trabalho pelo nó
intelligence down             # encerra o nó
```

`up` é idempotente. Execute-o novamente e ele informa que o nó já está em execução. A configuração fica em `~/.config/intelligence/config.toml`, o estado e os logs em `~/.local/state/intelligence`, e os modelos são procurados em `~/.local/share/intelligence/models` (as variáveis `XDG_*_HOME` são respeitadas).

Por padrão, o nó oferece três capacidades pequenas que funcionam em qualquer CPU: `inference.text`, `evaluation.text` e `training.reference`. O primeiro `infer` usa um classificador de sentimento embutido e retorna um rótulo, não uma resposta de chat. Isso é intencional: a instalação padrão deve funcionar em um Raspberry Pi sem arquivos de modelo e demonstrar o fluxo de roteamento, identidade e jobs antes de você configurar um modelo real. Veja [Modelos](#modelos) para usar llama.cpp.

Para testar a rede P2P em um laptop, inicie um segundo nó com sua própria configuração e conecte-o ao primeiro:

```bash
intelligence up
intelligence --config /tmp/node-b.toml up --peer 127.0.0.1:4000 --no-default-seeds
intelligence --config /tmp/node-b.toml network peers
intelligence --config /tmp/node-b.toml infer "this is great"
```

O nó B descobre o nó A, aprende suas capacidades e encaminha o job ao peer com as melhores evidências locais. Se o nó A parar, o nó B continua atendendo o que puder localmente.

Algo deu errado? `intelligence doctor` verifica configuração, identidade, permissões de armazenamento, endpoint QUIC, alcance por NAT, hardware detectado, modelos locais e saúde do nó, além de sugerir correções.

## O que você pode fazer

- **Descobrir peers sem um serviço de diretório.** Endereços bootstrap são dicas, não autoridades. Depois de encontrar um peer, o nó aprende sobre outros por troca assinada e por uma DHT limitada no estilo Kademlia, e lembra deles entre reinicializações.
- **Executar inferência entre máquinas.** Envie um job localmente; o nó o executa ou encaminha a um peer que anuncia a capacidade e conquistou confiança local para ela. Os resultados retornam em fluxo, com limites de tamanho e prazo.
- **Oferecer um modelo real à rede.** Configure uma capability `llama_cpp` com um binário `llama-cli` e um arquivo `.gguf` que você já possua. Jobs remotos executam dentro do bubblewrap.
- **Treinar entre máquinas.** Jobs de referência usam processos workers reais, sharding de parâmetros, estado de otimizador replicado, checkpoints content-addressed, agregação hierárquica e operações de referência tensor/pipeline-parallel. Nenhum coordenador único vê todas as atualizações, e nenhum worker precisa do modelo inteiro.
- **Continuar após falhas.** A suíte testa a interrupção de peers bootstrap, coordenadores, donos de shards, relays e estágios de pipeline. Reconexão, backoff, substituição de coordenador por termo, promoção de réplicas e retomada de checkpoints também são exercitados.
- **Combinar hardware.** Os nós anunciam backends de compute (CPU; CUDA, ROCm e Metal quando compilados com essas features), formatos e memória. O planner usa essas informações para posicionar shards e escolher estratégias.
- **Mover artefatos com segurança.** Modelos, checkpoints e datasets são identificados por conteúdo com BLAKE3, transferidos em blocos retomáveis, verificados ao chegar e colocados em quarentena se falharem na validação de integridade ou quota.
- **Manter o controle.** Cada capability é opcional, limitada e local ou pública. Nada é baixado implicitamente. Um nó funciona mesmo sem peers.

## Como funciona

Cada nó tem uma identidade Ed25519 persistente. Os peers se comunicam por QUIC vinculado a essa identidade. Assim, cada registro recebido é assinado pelo peer que o produziu: endereços, anúncios de capabilities, registros DHT, atualizações de treinamento e checkpoints.

Capabilities são declarações. A confiança é local e baseada em evidências: cada nó acompanha o que observou diretamente de cada peer, separadamente do que terceiros relataram, e só encaminha trabalho a peers que atendem à sua própria política. Não existe uma pontuação global de reputação.

Um job possui limites explícitos de entrada, saída, memória, CPU e prazo. O runtime o admite em uma fila limitada, executa com cancelamento e persiste seu estado para sobreviver a reinicializações.

O treinamento distribuído divide o estado do modelo em shards pertencentes a workers, agrupa workers sob agregadores para que o coordenador veja apenas agregados, replica estado do otimizador e checkpoints entre peers e alterna funções de coordenação por termos monotônicos quando um responsável desaparece.

```text
todas as atualizações -> um coordenador      NÃO
uma autoridade do estado do otimizador       NÃO
uma autoridade dos checkpoints               NÃO
barreira síncrona global                     NÃO
o modelo precisa caber em um worker           NÃO
```

## Instalação

### Binário de release

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
```

No Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.ps1 | iex
```

Fixe uma versão com `INTELLIGENCE_VERSION=v1.0.0`, ou espelhe os artefatos e configure `INTELLIGENCE_RELEASE_BASE_URL`. O script recusa checksums incorretos e arquivos compactados com caminhos inseguros.

### Compilar do código-fonte

Rust 1.85 ou mais recente.

```bash
git clone https://github.com/Cowde-Labs/intelligence-network
cd intelligence-network
cargo build --release -p intelligence-cli
./target/release/intelligence up
```

Ative a detecção de aceleradores com `--features cuda`, `--features rocm` ou `--features metal`. O nó consulta o driver e anuncia o backend, sem baixar SDK de fornecedor.

### Como serviço

```bash
intelligence service install    # serviço de usuário: systemd no Linux, launchd no macOS ou tarefa agendada no Windows; sem root
```

Para uma instalação de sistema com conta de serviço restrita, consulte [`infra/systemd`](infra/systemd).

## Execução

`intelligence up` inicia o nó em segundo plano e retorna quando ele está saudável. `intelligence run` executa o mesmo nó em primeiro plano, útil sob um supervisor ou durante depuração.

```bash
intelligence up --peer 203.0.113.10:4000 --peer 198.51.100.20:4000
intelligence up --no-default-seeds                 # usa somente os peers informados
intelligence up --listen-addr 0.0.0.0:4000         # aceita conexões de outras máquinas
intelligence up --capability inference.text        # ativa um subconjunto das capabilities embutidas
intelligence --config ./node.toml up               # inicia um segundo nó ou usa outro caminho de configuração
```

O endereço padrão de escuta é `127.0.0.1:4000`. Para aceitar conexões de outra máquina, associe o nó a um endereço alcançável e, se quiser anunciá-lo aos peers, configure `advertise_addr`. Um nó associado a loopback aceita endereços privados automaticamente; outros filtram esses endereços, a menos que `allow_private_addresses = true` esteja configurado, como em uma LAN.

As seeds padrão compiladas são `bootstrap-1.intelligence.network:4000` e `bootstrap-2.intelligence.network:4000`. São dicas substituíveis: se estiverem indisponíveis, o nó informa e continua. Use `--no-default-seeds` ou `INTELLIGENCE_DEFAULT_SEEDS=host:port,host:port` para substituí-las.

## Modelos

O nó nunca baixa pesos. Você fornece o arquivo — qualquer tipo é aceito —, e o nó calcula seu hash e o acompanha como artefato identificado por conteúdo.

```bash
intelligence model list
intelligence model add ~/models/llama-3.2-1b-q4.gguf
intelligence model add ./model.bin --identity local.mymodel.v1 --format opaque
intelligence artifact inspect --artifact <hash>
intelligence artifact fetch --peer <node-id> --artifact <hash>
```

Para executar um modelo, adicione uma capability `llama_cpp` à configuração. Você fornece o binário `llama-cli` e o caminho do modelo; o nó fornece sandbox, limites e roteamento:

```toml
[[capabilities]]
name = "inference.llama-cpp"
version = 1
public = true
accept_remote_jobs = true
kind = "llama_cpp"
program = "/usr/local/bin/llama-cli"
model_path = "/srv/models/llama-3.2-1b-q4.gguf"
args = ["--ctx-size", "2048"]
sandbox = "bubblewrap"
max_input_bytes = 65536
max_output_bytes = 65536
memory_bytes = 4294967296
cpu_millis = 30000
```

```bash
intelligence infer --capability inference.llama-cpp "Explique QUIC em um parágrafo"
intelligence infer --capability inference.llama-cpp --local-only "..."
```

Uma capability `llama_cpp` ou `process` pública que aceite jobs remotos precisa usar `sandbox = "bubblewrap"`; caso contrário, o nó rejeita a configuração. Use `public = false` para manter o modelo somente local. O adaptador llama.cpp é experimental: os limites, a validação e a fronteira de processo são testados, mas ainda não foram exercitados com uma ampla variedade de modelos e compilações.

## Referência da CLI

As opções globais são `--config <path>` (ou `INTELLIGENCE_CONFIG`) e `--json` para saída legível por máquina nos comandos compatíveis.

### Ciclo de vida

| Comando | Função |
|---|---|
| `up [--peer …] [--no-default-seeds] [--capability …] [--listen-addr …]` | Cria configuração e identidade se necessário e inicia o nó em segundo plano |
| `down` | Encerra o nó localmente |
| `run` | Inicia o nó em primeiro plano |
| `init` | Cria configuração e identidade sem iniciar o nó |
| `service install` | Instala um serviço por usuário |
| `version` | Mostra versões do software e do protocolo |

### Inspeção

| Comando | Função |
|---|---|
| `status` | Saúde, peers, jobs e contadores de recursos |
| `doctor` | Verifica rede, armazenamento, hardware, modelos e saúde do nó |
| `config` | Configuração efetiva após ambiente e valores padrão |
| `jobs` | Jobs persistidos e ativos |
| `identity` | ID e chave pública deste nó |
| `network peers` | Peers conhecidos |
| `network capabilities` | Capabilities anunciadas pelo nó |
| `network stats` | Tabela de roteamento e estatísticas dos registros |
| `trust inspect --subject <node-id>` | Decisão local de confiança baseada em evidências |

### Trabalho

| Comando | Função |
|---|---|
| `infer [TEXT] [...]` | Envia um job de inferência |
| `evaluate --text … --expected-label …` | Avalia uma amostra e registra evidências |
| `jobs cancel --job-id <id>` | Cancela um job em execução |

### Modelos e artefatos

| Comando | Função |
|---|---|
| `model list` | Lista modelos nos diretórios locais padrão |
| `model add <path> [...]` | Importa e calcula o hash de um modelo |
| `model register [...]` | Registro de modelo de nível mais baixo |
| `artifact inspect --artifact <hash>` | Inspeciona um artefato local |
| `artifact fetch --peer <node-id> --artifact <hash>` | Busca e verifica um artefato em um peer |

### Identidade e DHT

| Comando | Função |
|---|---|
| `identity rotate --new-path … [...]` | Rotaciona a identidade com uma transição assinada |
| `network publish [...]` | Publica um registro assinado |
| `network lookup [...]` | Busca um registro |
| `network find --key …` | Busca peers próximos de uma chave de roteamento |

### Treinamento

| Comando | Função |
|---|---|
| `train [...]` | Executa e aguarda um job de treinamento distribuído |
| `train start …` | Inicia o job e retorna imediatamente |
| `train plan --model-bytes … [...]` | Planeja um job entre peers conhecidos |
| `train replan --job-id … [...]` | Propõe outro grafo para um job em execução |
| `train activate --job-id …` | Ativa um grafo aprovado |
| `train status --job-id …` | Mostra o estado de um job distribuído |
| `train cancel --job-id …` | Cancela um job em execução |
| `train migrate --job-id … [...]` | Migra um shard para outro peer |
| `train replicate --workers a,b` | Replica estado de treinamento para peers selecionados |
| `train reconcile --worker … [...]` | Reconcilia dois branches compatíveis |
| `train reference [...]` | Executa o pequeno treinamento de referência síncrono |

`intelligence <comando> --help` mostra todas as opções. Nomes de comandos anteriores à versão 1.0 continuam disponíveis como aliases ocultos.

## Configuração avançada

`intelligence config` imprime a configuração efetiva. Cada campo pode ser definido em TOML ou substituído por uma variável `INTELLIGENCE_*`, como `INTELLIGENCE_LISTEN_ADDR`, `INTELLIGENCE_BOOTSTRAP`, `INTELLIGENCE_STORAGE_QUOTA_BYTES` e `INTELLIGENCE_RUNTIME_MAX_CONCURRENT_JOBS`. Há um exemplo comentado em [`config/node.toml.example`](config/node.toml.example).

**Rede:** `listen_addr`, `advertise_addr`, `bootstrap`, `allow_private_addresses`, `max_connections` e `peer_ttl_seconds`. Hole punching é habilitado por padrão e tem tentativas finitas; não há garantia contra NATs hostis.

**Relays:** qualquer nó pode operar como relay. Configure `relay_enabled`, `relay_max_sessions` e `relay_max_bytes`; clientes podem listar `relay_addresses` e definir `prefer_relay`. Relays encaminham envelopes autenticados opacos, sem ler jobs nem se tornar autoridades.

**DHT:** `dht_enabled`, `dht_k`, `dht_alpha` e `dht_max_records`. A DHT usa apenas peers diretos autenticados, não relays.

**Armazenamento:** `[storage] quota_bytes` e `max_artifact_bytes`. Artefatos que falham na verificação de integridade são colocados em quarentena, não apagados.

**Runtime:** `[runtime] max_queued_jobs`, `max_concurrent_jobs`, `max_input_bytes`, `max_output_bytes`, `default_timeout_ms`, `process_memory_bytes`, `process_cpu_seconds` e `work_dir`.

**Capabilities:** cada `[[capabilities]]` usa `builtin_text`, `builtin_training`, `process` ou `llama_cpp`, sandbox `trusted_local` ou `bubblewrap`, limites individuais e flags `public`/`accept_remote_jobs`. Metadados são incluídos em anúncios assinados e usados no planejamento; são declarações, avaliadas junto com as evidências observadas.

**Treinamento:** `training_memory_bytes` é o orçamento local de estado que o shard atribuído precisa respeitar. `training_window_delay_ms` injeta atraso de worker para experimentos; deixe-o em zero no uso normal.

## Arquitetura

Sete crates Rust principais com dependências em uma direção:

```text
protocol      tipos wire versionados, validação, manifests e mensagens de treino
storage       estado local, artefatos por conteúdo, quarentena e quotas
runtime       admissão, deadlines, cancelamento, limites de processos e bubblewrap
network       identidade Ed25519, QUIC, descoberta assinada, gossip, DHT e relays
intelligence  evidências, confiança local, planejamento de compute e algoritmos de treino
node          composição, persistência de jobs e socket administrativo local
cli           binário intelligence
```

O protocolo wire é versionado (atualmente 1.6) e cada parser voltado a peers tem limites e testes de fuzz. A CLI conversa com o node por um canal administrativo local: Unix socket no Linux/macOS ou named pipe no Windows; nada escuta em TCP.

As regras de projeto são: sem plano de controle central, infraestrutura obrigatória própria, camada econômica ou dependência de protocolo cuja indisponibilidade derrube a rede. Um emulador determinístico separado (`crates/emulator`) modela topologias de descoberta e treinamento com 100 a 100.000 peers lógicos, sem iniciar tantos processos. Os resultados são identificados como `EMULATED`.

Veja [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para mais detalhes.

## Segurança e limitações

Todo peer remoto é tratado como entrada não confiável. Identidades e registros são assinados, artefatos têm hash verificado, replay e equivocation são rejeitados, e trabalhos remotos são limitados por tamanho, fila, concorrência, memória, taxa e prazo. Execução remota de processos externos só é permitida dentro do bubblewrap. Consulte [docs/SECURITY.md](docs/SECURITY.md).

- **Validação de rede:** validado em uma máquina e em laboratório controlado, não na Internet pública. Implantação entre operadores independentes, NATs hostis reais e comportamento em escala pública ainda precisam de medições.
- **Escala do treinamento:** sharding, replicação, substituição de coordenador, paralelismo tensor/pipeline e reconciliação funcionam em processos reais com cargas CPU determinísticas e limitadas. Isso não é treinamento de modelos de fronteira.
- **Confiança:** é local, não global. O sistema não prova uma pessoa por identidade nem garante resistência Sybil além dos cenários testados.
- **Tolerância bizantina:** limitada. Agregação por mediana e validação de atualizações dificultam envenenamento, mas não constituem BFT.
- **Aceleradores:** são anunciados para planejamento; inferência GPU geral não é executada pelo node. A inferência GPU atual ocorre dentro do processo llama.cpp do operador.
- **Plataformas:** releases para Linux, macOS e Windows. Execução remota isolada por bubblewrap e serviço systemd são exclusivos de Linux. macOS/Windows são verificados em CI, ainda sem implantações prolongadas.

## Problemas que precisamos resolver

Estes são pontos de trabalho identificados no estado atual do repositório, não funcionalidades já implementadas:

- Criar perfil local confiável de recursos e orçamento adaptativo, considerando limites de containers, carga, memória disponível, energia e temperatura.
- Fazer o scheduler comparar custo computacional, memória por shard, latência, largura de banda, transferência de dados e confiabilidade observada.
- Substituir estimativas e declarações de hardware insuficientes por observações atualizadas e explicáveis, sem publicar dados locais sensíveis.
- Ampliar execução real de aceleradores; detecção/anúncio de backend não deve ser confundida com kernels de produção.
- Validar segurança, conectividade e comportamento sob falhas em mais sistemas, redes e hardware heterogêneo.
- Preparar participação mobile oportunista sem presumir execução contínua em segundo plano.

## Desenvolvimento

```bash
export CARGO_TARGET_DIR=.cache/rust-target
cargo build -p intelligence-cli
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
```

Cenários de integração com processos node reais:

```bash
./scripts/compatibility-smoke.sh
./scripts/local-testnet.sh
```

O laboratório de rede controlado, emuladores de escala, fuzzing e pipeline de release estão descritos em [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Contribuições devem tornar o protocolo mais claro, interoperável ou resiliente a falhas. Parsers expostos a peers precisam de limites e testes para entradas malformadas; jobs longos precisam de cancelamento e recuperação; declarações de capacidade usadas no roteamento precisam de um caminho para evidências. Não adicione tokens, blockchains, marketplaces ou serviços centrais sem um ADR que altere a tese do projeto. Consulte [CONTRIBUTING.md](CONTRIBUTING.md).

## Licença

GNU Affero General Public License v3.0 somente. Consulte [LICENSE](LICENSE).

---

Esta tradução é mantida pela comunidade e pode ficar atrás do original. Em caso de divergência, prevalece [README.md](README.md).
