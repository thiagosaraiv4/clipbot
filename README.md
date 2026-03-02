# ◀ ClipBot — App Desktop

Aplicativo com interface gráfica para gerar e postar clips de lives automaticamente.
Roda 24h em background, minimizado na bandeja do sistema.

---

## ✅ Como instalar e rodar

### Windows (mais fácil)

**Opção A — Executar do código-fonte:**
```
1. Instale Python 3.11+  →  python.org/downloads
2. Instale ffmpeg        →  ffmpeg.org/download.html (adicione ao PATH)
3. Abra o terminal nessa pasta e rode:
   pip install -r requirements.txt
   python app.py
```

**Opção B — Gerar .exe para distribuir:**
```
1. Faça os passos acima
2. Dê duplo clique em:  build_windows.bat
3. O executável fica em:  ClipBot-Windows\ClipBot.exe
```

### macOS / Linux
```bash
chmod +x build_mac_linux.sh
./build_mac_linux.sh
# Executável em ClipBot-dist/
```

---

## 🖥 Como usar o app

### 1. Aba "🎥 Lives" — Configure as plataformas
- Ative os toggles das plataformas que você usa
- Digite o nome do canal de cada uma
- Escolha a pasta onde salvar os clips

### 2. Aba "🔑 Contas" — Faça login
- **TikTok:** Cole o `sessionid` do cookie do navegador
- **Instagram:** Digite usuário e senha
- **YouTube:** Selecione o arquivo `youtube_credentials.json`
- Clique em **SALVAR CREDENCIAIS**

### 3. Aba "⚙️ Detecção" — Ajuste a sensibilidade
- Ative os métodos desejados (áudio, chat, palavras-chave)
- Ajuste os sliders de sensibilidade
- Configure as palavras-chave para o seu chat

### 4. Aba "✂️ Edição" — Configure os clips
- Escolha vertical (TikTok/Reels/Shorts) ou horizontal
- Ative legendas automáticas (IA Whisper — grátis, offline)
- Defina limite de posts por dia

### 5. Clique em ▶ INICIAR
O bot começa a monitorar as lives e postar automaticamente!

---

## 📌 Rodar 24 horas

Ao fechar a janela, o app **vai para a bandeja do sistema** (canto inferior direito no Windows).
Clique no ícone ◀ para abrir novamente.

Para iniciar automaticamente com o Windows:
- Na aba "✂️ Edição" → Sistema → ative **"Iniciar bot automaticamente ao abrir o app"**
- Adicione o `ClipBot.exe` à pasta de inicialização do Windows:
  `Win + R` → `shell:startup` → cole um atalho do ClipBot.exe

---

## 🔑 Como pegar o Session ID do TikTok

1. Abra tiktok.com no Chrome e faça login
2. Pressione `F12` → aba **Application**
3. No menu lateral: **Cookies** → `https://www.tiktok.com`
4. Encontre o cookie **`sessionid`**
5. Copie o valor e cole no ClipBot

---

## 📁 Arquivos gerados

Os clips e dados ficam em:
- **Windows:** `C:\Users\SEU_USUARIO\AppData\Roaming\ClipBot\`
- **Mac:** `~/Library/Application Support/ClipBot/`
- **Linux:** `~/.config/ClipBot/`

---

## ❓ Problemas comuns

| Problema | Solução |
|---|---|
| "ffmpeg não encontrado" | Instale o ffmpeg e adicione ao PATH |
| TikTok não posta | Session ID expirado — copie novamente do navegador |
| Instagram bloqueado | Desative 2FA temporariamente na primeira vez |
| YouTube pede autorização | Normal — autorize pelo navegador uma única vez |
| Bot para sozinho | Verifique se a live ainda está ativa |
