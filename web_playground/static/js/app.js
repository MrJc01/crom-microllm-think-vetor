// STATE MANAGEMENT
let isGenerating = false;
let activeController = null;
let systemStatsInterval = null;
let chatHistoryList = [];

// Resource chart variables
const resourceHistory = {
    cpu: [],
    ram: [],
    labels: []
};
const MAX_HISTORY_LEN = 20;

// UI ELEMENTS
const chatHistory = document.getElementById('chat-history');
const welcomeContainer = document.getElementById('welcome-container');
const promptInput = document.getElementById('prompt-input');
const btnSend = document.getElementById('btn-send');
const btnStop = document.getElementById('btn-stop');
const btnClearChat = document.getElementById('btn-clear-chat');
const toggleRightSidebar = document.getElementById('toggle-right-sidebar');
const sidebarRight = document.getElementById('sidebar-right');

// Config parameters
const modelSelect = document.getElementById('model-select');
const modelStatusText = document.getElementById('model-status-text');
const modelStatusIndicator = document.querySelector('.status-indicator');
const paramTemp = document.getElementById('param-temp');
const tempVal = document.getElementById('temp-val');
const paramMaxTokens = document.getElementById('param-max-tokens');
const maxTokensVal = document.getElementById('max-tokens-val');
const paramTvDsl = document.getElementById('param-tv-dsl');

// System stats elements
const cpuStat = document.getElementById('cpu-stat');
const cpuProgress = document.getElementById('cpu-progress');
const ramStat = document.getElementById('ram-stat');
const ramProgress = document.getElementById('ram-progress');
const deviceType = document.getElementById('device-type');

// Diagnostics elements
const valTtft = document.getElementById('val-ttft');
const valLatency = document.getElementById('val-latency');
const valSpeed = document.getElementById('val-speed');
const valTokens = document.getElementById('val-tokens');
const dslBadge = document.getElementById('dsl-badge');
const dslConsole = document.getElementById('dsl-console');
const entropyAvgVal = document.getElementById('entropy-avg-val');

// CANVAS RESOURCE CHART
const resourcesChart = document.getElementById('resources-chart');
const ctxResources = resourcesChart.getContext('2d');

// INITIALIZE APP
document.addEventListener('DOMContentLoaded', () => {
    // Sync parameter labels
    paramTemp.addEventListener('input', (e) => tempVal.textContent = e.target.value);
    paramMaxTokens.addEventListener('input', (e) => maxTokensVal.textContent = e.target.value);
    
    // Auto-resize input textarea
    promptInput.addEventListener('input', adjustTextareaHeight);
    promptInput.addEventListener('keydown', handleEnterKey);
    
    // Event listeners
    btnSend.addEventListener('click', sendMessage);
    btnStop.addEventListener('click', stopGeneration);
    btnClearChat.addEventListener('click', clearChat);
    toggleRightSidebar.addEventListener('click', toggleSidebar);
    
    // Initialize suggestions cards
    document.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            promptInput.value = card.getAttribute('data-prompt');
            adjustTextareaHeight();
            sendMessage();
        });
    });
    
    // Model Selection change
    modelSelect.addEventListener('change', changeActiveModel);
    
    // Fetch models and system stats
    fetchModels();
    updateSystemStats();
    systemStatsInterval = setInterval(updateSystemStats, 3000);
    
    // Resize Canvas on start
    resizeChartCanvas();
    window.addEventListener('resize', () => {
        resizeChartCanvas();
        drawChart();
    });
});

// AUTO-RESIZE INPUT TEXTAREA
function adjustTextareaHeight() {
    promptInput.style.height = 'auto';
    promptInput.style.height = (promptInput.scrollHeight) + 'px';
}

function handleEnterKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// TOGGLE DIAGNOSTICS SIDEBAR
function toggleSidebar() {
    sidebarRight.classList.toggle('collapsed');
    const isCollapsed = sidebarRight.classList.contains('collapsed');
    toggleRightSidebar.querySelector('i').className = isCollapsed ? 'fa-solid fa-chart-column text-muted' : 'fa-solid fa-chart-column';
    toggleRightSidebar.title = isCollapsed ? 'Mostrar diagnósticos' : 'Ocultar diagnósticos';
    setTimeout(() => {
        resizeChartCanvas();
        drawChart();
    }, 310); // Wait for transition
}

// LOAD AVAILABLE MODELS
async function fetchModels() {
    try {
        modelStatusIndicator.className = "status-indicator loading";
        modelStatusText.textContent = "Buscando modelos...";
        
        const response = await fetch('/api/models');
        const data = await response.json();
        
        modelSelect.innerHTML = '';
        if (data.models.length === 0) {
            modelSelect.innerHTML = '<option value="">Nenhum modelo em checkpoints/</option>';
            modelStatusIndicator.className = "status-indicator offline";
            modelStatusText.textContent = "Sem modelos";
            return;
        }
        
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = model.name;
            if (model.path === data.active_model) {
                option.selected = true;
            }
            modelSelect.appendChild(option);
        });
        
        if (data.active_model) {
            modelStatusIndicator.className = "status-indicator online";
            modelStatusText.textContent = "Pronto";
        } else {
            modelStatusIndicator.className = "status-indicator offline";
            modelStatusText.textContent = "Nenhum carregado";
        }
    } catch (e) {
        console.error("Erro ao buscar modelos:", e);
        modelStatusIndicator.className = "status-indicator offline";
        modelStatusText.textContent = "Erro de conexão";
    }
}

// CHANGE SELECTED MODEL
async function changeActiveModel() {
    const selectedPath = modelSelect.value;
    if (!selectedPath) return;
    
    try {
        isGenerating = false;
        btnSend.disabled = true;
        modelStatusIndicator.className = "status-indicator loading";
        modelStatusText.textContent = "Carregando pesos...";
        
        const response = await fetch('/api/select_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: selectedPath })
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            modelStatusIndicator.className = "status-indicator online";
            modelStatusText.textContent = "Pronto";
        } else {
            modelStatusIndicator.className = "status-indicator offline";
            modelStatusText.textContent = "Falha: " + data.message;
        }
    } catch (e) {
        console.error("Erro ao alterar modelo:", e);
        modelStatusIndicator.className = "status-indicator offline";
        modelStatusText.textContent = "Erro na alteração";
    } finally {
        btnSend.disabled = false;
    }
}

// FETCH SYSTEM STATS
async function updateSystemStats() {
    try {
        const response = await fetch('/api/system_stats');
        const data = await response.json();
        
        cpuStat.textContent = `${data.cpu_usage}%`;
        cpuProgress.style.width = `${data.cpu_usage}%`;
        
        ramStat.textContent = `${data.ram_usage}%`;
        ramProgress.style.width = `${data.ram_usage}%`;
        
        deviceType.textContent = data.device;
        
        // Update resources history graph
        const now = new Date();
        const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
        
        resourceHistory.cpu.push(data.cpu_usage);
        resourceHistory.ram.push(data.ram_usage);
        resourceHistory.labels.push(timeStr);
        
        if (resourceHistory.cpu.length > MAX_HISTORY_LEN) {
            resourceHistory.cpu.shift();
            resourceHistory.ram.shift();
            resourceHistory.labels.shift();
        }
        
        drawChart();
    } catch (e) {
        console.error("Erro ao atualizar status do sistema:", e);
    }
}

// CANVAS DRAW CHART FUNCTIONS
function resizeChartCanvas() {
    const rect = resourcesChart.getBoundingClientRect();
    resourcesChart.width = rect.width;
    resourcesChart.height = rect.height;
}

function drawChart() {
    const w = resourcesChart.width;
    const h = resourcesChart.height;
    
    // Clear canvas
    ctxResources.clearRect(0, 0, w, h);
    
    // Draw background grid lines (2 horizontal lines)
    ctxResources.strokeStyle = '#222936';
    ctxResources.lineWidth = 1;
    ctxResources.beginPath();
    ctxResources.moveTo(0, h * 0.33); ctxResources.lineTo(w, h * 0.33);
    ctxResources.moveTo(0, h * 0.66); ctxResources.lineTo(w, h * 0.66);
    ctxResources.stroke();
    
    const len = resourceHistory.cpu.length;
    if (len < 2) return;
    
    const step = w / (MAX_HISTORY_LEN - 1);
    
    // Helper function to plot a line
    const plotLine = (data, color) => {
        ctxResources.strokeStyle = color;
        ctxResources.lineWidth = 2;
        ctxResources.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = i * step;
            const y = h - (data[i] / 100 * (h - 10)) - 5;
            if (i === 0) ctxResources.moveTo(x, y);
            else ctxResources.lineTo(x, y);
        }
        ctxResources.stroke();
    };
    
    // Draw RAM (purple)
    plotLine(resourceHistory.ram, '#8ab4f8');
    // Draw CPU (blue)
    plotLine(resourceHistory.cpu, '#1a73e8');
}

// CLEAR CONVERSATION
function clearChat() {
    chatHistoryList = [];
    chatHistory.innerHTML = '';
    welcomeContainer.classList.remove('hidden');
    chatHistory.appendChild(welcomeContainer);
    
    // Reset diagnostics display
    valTtft.textContent = '-';
    valLatency.textContent = '-';
    valSpeed.textContent = '-';
    valTokens.textContent = '-';
    dslBadge.className = 'badge badge-none';
    dslBadge.textContent = 'Nenhum';
    dslConsole.textContent = 'Aguardando chamada de equações no raciocínio...';
    
    document.querySelectorAll('.entropy-bar-fill').forEach(bar => {
        bar.style.height = '0%';
    });
    document.querySelectorAll('.entropy-bar-value').forEach(val => {
        val.textContent = '0.0';
    });
    entropyAvgVal.textContent = '-';
}

// STOP ACTIVE GENERATION
function stopGeneration() {
    if (activeController) {
        activeController.abort();
        activeController = null;
        isGenerating = false;
        
        btnStop.classList.add('hidden');
        btnSend.classList.remove('hidden');
        promptInput.disabled = false;
        
        // Remove typing indicators if any
        document.querySelectorAll('.typing-indicator').forEach(el => el.remove());
    }
}

// SEND MESSAGE VIA STREAM
async function sendMessage() {
    const text = promptInput.value.trim();
    if (!text || isGenerating) return;
    
    isGenerating = true;
    welcomeContainer.classList.add('hidden');
    
    // Disable inputs
    promptInput.disabled = true;
    promptInput.value = '';
    adjustTextareaHeight();
    
    // Toggle send button to stop button
    btnSend.classList.add('hidden');
    btnStop.classList.remove('hidden');
    
    // 1. Render User Message
    appendMessage(text, 'user');
    
    // 2. Render Assistant Message container
    const messageEl = document.createElement('div');
    messageEl.className = 'chat-message assistant';
    
    const avatarEl = document.createElement('div');
    avatarEl.className = 'avatar';
    avatarEl.innerHTML = '<i class="fa-solid fa-brain"></i>';
    
    const bodyEl = document.createElement('div');
    bodyEl.className = 'message-body';
    
    const senderEl = document.createElement('div');
    senderEl.className = 'message-sender';
    senderEl.textContent = 'Think-Vetor';
    
    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';
    
    // Typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    contentEl.appendChild(typingIndicator);
    
    bodyEl.appendChild(senderEl);
    bodyEl.appendChild(contentEl);
    messageEl.appendChild(avatarEl);
    messageEl.appendChild(bodyEl);
    chatHistory.appendChild(messageEl);
    scrollChatToBottom();
    
    // Reset Diagnostics side-panels for new generation
    valTtft.textContent = '...';
    valLatency.textContent = '...';
    valSpeed.textContent = '...';
    valTokens.textContent = '...';
    dslBadge.className = 'badge badge-none';
    dslBadge.textContent = 'Buscando...';
    dslConsole.textContent = 'Escutando chamada de equações...';
    
    // Stream states
    let thoughtContainer = null;
    let thoughtBody = null;
    let currentBlockIsThinking = false;
    let fullRawResponse = "";
    
    activeController = new AbortController();
    const signal = activeController.signal;
    
    // Registrar mensagem do usuário no histórico local
    chatHistoryList.push({ role: "user", content: text });
    
    const requestPayload = {
        history: chatHistoryList,
        model_path: modelSelect.value,
        temperature: parseFloat(paramTemp.value),
        max_new_tokens: parseInt(paramMaxTokens.value),
        use_tv_dsl: paramTvDsl.checked
    };
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestPayload),
            signal: signal
        });
        
        if (!response.ok) {
            throw new Error(`Erro na conexão HTTP: ${response.status}`);
        }
        
        // Remove typing indicator
        typingIndicator.remove();
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let streamBuffer = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            streamBuffer += decoder.decode(value, { stream: true });
            
            // Process SSE rows
            const lines = streamBuffer.split('\n');
            // Keep the last incomplete line in buffer
            streamBuffer = lines.pop();
            
            for (const line of lines) {
                const cleanLine = line.trim();
                if (!cleanLine.startsWith('data: ')) continue;
                
                const eventDataStr = cleanLine.substring(6);
                if (eventDataStr === '[DONE]') continue;
                
                try {
                    const event = JSON.parse(eventDataStr);
                    
                    // Type 1: Token chunk
                    if (event.type === 'token') {
                        const token = event.token;
                        fullRawResponse += token;
                        
                        // Check if we need to open a thinking block
                        if (event.is_thinking && !currentBlockIsThinking) {
                            currentBlockIsThinking = true;
                            
                            thoughtContainer = document.createElement('div');
                            thoughtContainer.className = 'thought-container';
                            
                            const header = document.createElement('div');
                            header.className = 'thought-header';
                            header.innerHTML = `
                                <span class="thought-title"><i class="fa-solid fa-lightbulb"></i> Pensamento Latente Cognitivo</span>
                                <i class="fa-solid fa-chevron-down thought-icon"></i>
                            `;
                            
                            thoughtBody = document.createElement('div');
                            thoughtBody.className = 'thought-body';
                            
                            thoughtContainer.appendChild(header);
                            thoughtContainer.appendChild(thoughtBody);
                            contentEl.appendChild(thoughtContainer);
                            
                            // Collapsible interaction
                            header.addEventListener('click', () => {
                                thoughtContainer.classList.toggle('collapsed');
                            });
                        }
                        
                        // Append token
                        if (currentBlockIsThinking) {
                            // Filter out tags in thought block display
                            let cleanToken = token.replace('<thought>', '').replace('</thought>', '');
                            if (cleanToken) {
                                thoughtBody.textContent += cleanToken;
                            }
                        } else {
                            // Main response stream
                            let cleanToken = token.replace('<thought>', '').replace('</thought>', '');
                            // Filter out chat formatting tags if returned
                            cleanToken = cleanToken.replace('<|im_end|>', '');
                            
                            if (cleanToken) {
                                contentEl.appendChild(document.createTextNode(cleanToken));
                            }
                        }
                        
                        // If token tells us thinking is complete, switch state
                        if (!event.is_thinking && currentBlockIsThinking) {
                            currentBlockIsThinking = false;
                        }
                        
                        // Update diagnostics sidebar gauges
                        valTtft.textContent = event.ttft_ms;
                        valLatency.textContent = event.elapsed_ms;
                        valSpeed.textContent = event.tokens_sec;
                        valTokens.textContent = event.total_tokens;
                        
                        // Update attention head entropy
                        if (event.entropy) {
                            updateEntropyViz(event.entropy);
                        }
                        
                        scrollChatToBottom();
                    }
                    
                    // Type 2: TV-DSL Intercept event
                    else if (event.type === 'tv_dsl') {
                        // Display inside chat as AST visual bubble
                        const dslContainer = document.createElement('div');
                        dslContainer.className = 'dsl-call-highlight';
                        dslContainer.innerHTML = `
                            <strong><i class="fa-solid fa-terminal text-warning"></i> AST Coprocessor Intercept:</strong><br>
                            ${event.message}
                        `;
                        contentEl.appendChild(dslContainer);
                        
                        // Log to diagnostics console
                        dslConsole.textContent = `RAW:\n${event.raw}\n\nPROCESSED:\n${event.processed}`;
                        dslBadge.className = 'badge badge-valid';
                        dslBadge.textContent = 'Executado';
                        
                        // O coprocessador TV-DSL altera o prompt, vamos adicionar a resposta parcial no fullRawResponse
                        fullRawResponse = event.processed + "\n";
                        
                        scrollChatToBottom();
                    }
                    
                    // Type 3: Generation completed
                    else if (event.type === 'done') {
                        valTtft.textContent = event.ttft_ms;
                        valLatency.textContent = event.elapsed_ms;
                        valSpeed.textContent = event.avg_tokens_sec;
                        valTokens.textContent = event.total_tokens;
                        
                        if (event.has_tv_dsl) {
                            if (event.tv_dsl_valid) {
                                dslBadge.className = 'badge badge-valid';
                                dslBadge.textContent = 'Válido';
                            } else {
                                dslBadge.className = 'badge badge-invalid';
                                dslBadge.textContent = 'Falha AST';
                            }
                        } else {
                            dslBadge.className = 'badge badge-none';
                            dslBadge.textContent = 'Nenhum';
                        }
                    }
                    
                } catch (pe) {
                    console.error("Erro ao fazer parse da linha SSE:", pe, cleanLine);
                }
            }
        }
        
        // Registrar a resposta final do assistente no histórico
        if (fullRawResponse) {
            chatHistoryList.push({ role: "assistant", content: fullRawResponse });
        }
        
    } catch (err) {
        // Remover última mensagem do usuário do histórico em caso de erro/abortagem
        if (chatHistoryList.length > 0) {
            chatHistoryList.pop();
        }
        
        if (err.name === 'AbortError') {
            const stopNotice = document.createElement('div');
            stopNotice.className = 'text-muted';
            stopNotice.style.fontSize = '0.85rem';
            stopNotice.style.fontStyle = 'italic';
            stopNotice.style.marginTop = '10px';
            stopNotice.textContent = 'Geração interrompida pelo usuário.';
            contentEl.appendChild(stopNotice);
        } else {
            console.error("Erro na geração:", err);
            const errNotice = document.createElement('div');
            errNotice.className = 'text-danger';
            errNotice.style.fontSize = '0.85rem';
            errNotice.style.marginTop = '10px';
            errNotice.textContent = `Erro de conexão: ${err.message}`;
            contentEl.appendChild(errNotice);
        }
    } finally {
        isGenerating = false;
        activeController = null;
        
        btnStop.classList.add('hidden');
        btnSend.classList.remove('hidden');
        promptInput.disabled = false;
        
        scrollChatToBottom();
    }
}

// UPDATE ATTENTION ENTROPY GAUGE
function updateEntropyViz(entropyList) {
    if (!entropyList || entropyList.length === 0) return;
    
    let sum = 0;
    const bars = document.querySelectorAll('.entropy-bar-fill');
    const labels = document.querySelectorAll('.entropy-bar-value');
    
    entropyList.forEach((val, idx) => {
        if (idx >= bars.length) return;
        
        sum += val;
        
        // Calculate percentage height relative to max entropy (approx 5.5)
        const pct = Math.max(0, Math.min(100, (val / 5.5) * 100));
        bars[idx].style.height = `${pct}%`;
        labels[idx].textContent = val.toFixed(1);
        
        // Dynamic coloring of bar according to heat (hotter = more red/orange, cooler = purple/blue)
        if (val > 4.6) {
            bars[idx].style.background = 'linear-gradient(to top, #ea4335, #f9ab00)'; // Hot
        } else if (val > 4.25) {
            bars[idx].style.background = 'linear-gradient(to top, #1a73e8, #f9ab00)'; // Med-hot
        } else {
            bars[idx].style.background = 'linear-gradient(to top, #1a73e8, #8ab4f8)'; // Cool
        }
    });
    
    const avg = sum / entropyList.length;
    entropyAvgVal.textContent = avg.toFixed(3);
}

// AUX: APPEND MESSAGE
function appendMessage(text, sender) {
    const messageEl = document.createElement('div');
    messageEl.className = `chat-message ${sender}`;
    
    const avatarEl = document.createElement('div');
    avatarEl.className = 'avatar';
    avatarEl.innerHTML = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-brain"></i>';
    
    const bodyEl = document.createElement('div');
    bodyEl.className = 'message-body';
    
    const senderEl = document.createElement('div');
    senderEl.className = 'message-sender';
    senderEl.textContent = sender === 'user' ? 'Você' : 'Think-Vetor';
    
    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';
    contentEl.textContent = text;
    
    bodyEl.appendChild(senderEl);
    bodyEl.appendChild(contentEl);
    messageEl.appendChild(avatarEl);
    messageEl.appendChild(bodyEl);
    chatHistory.appendChild(messageEl);
    scrollChatToBottom();
}

// AUX: SCROLL TO BOTTOM
function scrollChatToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}
