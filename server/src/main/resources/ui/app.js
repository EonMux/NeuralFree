/*
 * NeuralFree chat UI: the application.
 *
 * State lives in localStorage (conversations, settings); the server endpoints
 * (/health, /props, /v1/models, /slots, /v1/chat/completions) carry every model
 * interaction. Streaming uses the OpenAI server-sent event form, including the
 * reasoning_content deltas. No dependencies.
 */
window.NfApp = (function () {
  'use strict';

  const CONVERSATIONS_KEY = 'nf.conversations.v1';
  const SETTINGS_KEY = 'nf.settings.v1';

  const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/svg+xml'];
  const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'];
  const TEXT_EXTENSIONS = ['txt', 'md', 'markdown', 'json', 'csv', 'log', 'py', 'js', 'ts', 'tsx', 'jsx',
    'java', 'c', 'h', 'cc', 'cpp', 'hpp', 'rs', 'go', 'sh', 'bash', 'yaml', 'yml', 'toml', 'html', 'css',
    'xml', 'sql', 'ini', 'cfg', 'conf', 'properties', 'gradle', 'kt', 'rb', 'php', 'lua', 'diff', 'patch'];
  const IMAGE_LIMIT = 4 * 1024 * 1024;
  const TEXT_LIMIT = 200 * 1024;
  const TITLE_LIMIT = 48;

  const md = () => window.NfMarkdown;

  // ---------- state ----------

  let conversations = [];
  let currentId = null;
  let settings = defaultSettings();
  let attachments = [];
  let stream = null;          // { controller, messageEl, bodyEl, text, reasoning, dirty, timer }
  let slotsTimer = null;

  function defaultSettings() {
    return {
      apiKey: '', theme: 'dark', defaultSystem: '', model: '',
      temperature: '', topP: '', maxTokens: '', seed: ''
    };
  }

  function loadJson(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (e) {
      console.warn('stored ' + key + ' does not read: ' + e);
      return fallback;
    }
  }

  function saveConversations() {
    try {
      localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
    } catch (e) {
      showError('conversations do not fit the browser storage: ' + e);
    }
  }

  function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }

  function current() {
    return conversations.find((c) => c.id === currentId) || null;
  }

  function newId() {
    return (crypto && crypto.randomUUID) ? crypto.randomUUID() :
      'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
  }

  // ---------- api ----------

  /**
   * The base the API and UI assets sit at: the prefix of the document's own script URL, so under an --api-prefix
   * the endpoints answer beside the document there and at the root they answer at the root.
   */
  const BASE = (function () {
    const mark = '/ui/app.js';
    for (const script of document.querySelectorAll('script[src]')) {
      const path = new URL(script.src).pathname;
      if (path.endsWith(mark)) {
        return path.slice(0, path.length - mark.length);
      }
    }
    return '';
  })();

  function headers(json) {
    const out = {};
    if (json) out['Content-Type'] = 'application/json';
    if (settings.apiKey) out['Authorization'] = 'Bearer ' + settings.apiKey;
    return out;
  }

  async function errorOf(response) {
    let message = 'HTTP ' + response.status;
    try {
      const body = await response.json();
      if (body && body.error && body.error.message) message = body.error.message;
    } catch (e) { /* the status alone then */ }
    const error = new Error(message);
    error.status = response.status;
    return error;
  }

  async function api(path, options) {
    const response = await fetch(BASE + path, {
      method: (options && options.method) || 'GET',
      headers: headers(Boolean(options && options.body)),
      body: options && options.body
    });
    if (!response.ok) {
      throw await errorOf(response);
    }
    return response.json();
  }

  // ---------- server status ----------

  let serverProps = null;
  let routerProps = null;
  let propsError = null;
  let models = [];

  async function refreshServer() {
    setStatus('off', 'checking the server');
    const [health, base, list] = await Promise.allSettled([
      api('/health'), api('/props'), api('/v1/models')
    ]);
    if (health.status === 'fulfilled') {
      const ok = String(health.value && health.value.status || '').toLowerCase() === 'ok';
      setStatus(ok ? 'ok' : 'down', 'server ' + (health.value && health.value.status || 'unknown'));
    } else {
      setStatus('down', health.reason && health.reason.message || 'the server does not answer');
    }
    routerProps = base.status === 'fulfilled' ? base.value : null;
    models = list.status === 'fulfilled' && Array.isArray(list.value.data) ? list.value.data : [];
    applyServerDefaults();
    renderModelSelect();
    await refreshProps();
    if (document.getElementById('server-dialog').open) {
      renderServerInfo();
    }
  }

  /** Whether the server serves several models and routes every request by its model field. */
  function routerMode() {
    return Boolean(routerProps && routerProps.role === 'router');
  }

  /** The properties of the selected model, or the router's own while no model is selected. */
  async function refreshProps() {
    const model = selectedModel();
    const path = model
      ? '/props?model=' + encodeURIComponent(model) + (routerMode() ? '&autoload=false' : '')
      : '/props';
    try {
      serverProps = await api(path);
      propsError = null;
    } catch (error) {
      serverProps = null;
      propsError = error;
    }
    updateModelLoad();
  }

  function setStatus(state, title) {
    const dot = document.getElementById('server-status');
    dot.className = 'status-dot ' + state;
    dot.title = title;
  }

  /**
   * The server's --ui-config as the defaults of the settings the user has not stored yet: the server carries them
   * in /props ui_settings, and anything in localStorage keeps winning.
   */
  function applyServerDefaults() {
    const config = serverProps && serverProps.ui_settings;
    if (!config || typeof config !== 'object' || Array.isArray(config)) {
      return;
    }
    const stored = loadJson(SETTINGS_KEY, {});
    for (const key of Object.keys(defaultSettings())) {
      if (config[key] !== undefined && stored[key] === undefined) {
        settings[key] = config[key];
      }
    }
    applyTheme();
  }

  /** A model's load state: the status object of a router listing, or empty outside router mode. */
  function modelState(model) {
    if (!model || !model.status) return '';
    const status = typeof model.status === 'object' ? model.status : { value: model.status };
    if (status.failed) return 'failed';
    return String(status.value || '');
  }

  function selectedEntry() {
    const id = selectedModel();
    return models.find((model) => model.id === id) || null;
  }

  /** Shows the load action while the router lists the selected model as unloaded or failed. */
  function updateModelLoad() {
    const button = document.getElementById('model-load');
    if (!button) return;
    const entry = selectedEntry();
    const state = modelState(entry);
    const loadable = routerMode() && entry && (state === 'unloaded' || state === 'failed');
    button.hidden = !loadable;
    if (!loadable) {
      button.title = '';
    } else {
      const status = typeof entry.status === 'object' ? entry.status : {};
      button.title = status.error ? 'Load ' + entry.id + ': ' + status.error : 'Load ' + entry.id;
    }
  }

  function renderModelSelect() {
    const select = document.getElementById('model-select');
    select.innerHTML = '';
    if (!models.length) {
      const option = document.createElement('option');
      option.textContent = serverProps || routerProps ? 'no model listed' : 'server unreachable';
      option.disabled = true;
      option.selected = true;
      select.appendChild(option);
      updateModelLoad();
      return;
    }
    for (const model of models) {
      const option = document.createElement('option');
      option.value = model.id;
      const state = routerMode() ? modelState(model) : '';
      option.textContent = state ? model.id + ' — ' + state : model.id;
      select.appendChild(option);
    }
    const wanted = settings.model;
    select.value = models.some((m) => m.id === wanted) ? wanted : models[0].id;
    updateModelLoad();
  }

  async function loadSelectedModel() {
    const entry = selectedEntry();
    if (!entry) return;
    const button = document.getElementById('model-load');
    button.disabled = true;
    try {
      await api('/models/load', { method: 'POST', body: JSON.stringify({ model: entry.id }) });
      await refreshServer();
    } catch (error) {
      showError('the model does not load: ' + error.message);
    } finally {
      button.disabled = false;
    }
  }

  function selectedModel() {
    return document.getElementById('model-select').value || '';
  }

  // ---------- conversations ----------

  function createConversation(select) {
    const conversation = {
      id: newId(),
      title: 'New chat',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: []
    };
    conversations.unshift(conversation);
    if (select) {
      currentId = conversation.id;
    }
    saveConversations();
    renderSidebar();
    renderMessages();
    document.getElementById('input').focus();
    return conversation;
  }

  function deleteConversation(id) {
    const conversation = conversations.find((c) => c.id === id);
    if (!conversation || !window.confirm('Delete "' + conversation.title + '"?')) {
      return;
    }
    conversations = conversations.filter((c) => c.id !== id);
    if (currentId === id) {
      currentId = conversations.length ? conversations[0].id : null;
      if (!currentId) {
        createConversation(false);
      }
    }
    saveConversations();
    renderSidebar();
    renderMessages();
  }

  function renameConversation(id) {
    const conversation = conversations.find((c) => c.id === id);
    const name = window.prompt('Conversation name', conversation ? conversation.title : '');
    if (!conversation || name === null) {
      return;
    }
    conversation.title = name.trim() || conversation.title;
    conversation.updatedAt = Date.now();
    saveConversations();
    renderSidebar();
  }

  function autoTitle(conversation) {
    if (conversation.title !== 'New chat') {
      return;
    }
    const first = conversation.messages.find((m) => m.role === 'user');
    if (first && first.content) {
      const text = String(first.content).replace(/\s+/g, ' ').trim();
      conversation.title = text.slice(0, TITLE_LIMIT) + (text.length > TITLE_LIMIT ? '…' : '');
    }
  }

  function renderSidebar() {
    const list = document.getElementById('conversation-list');
    list.innerHTML = '';
    const query = document.getElementById('search').value.trim().toLowerCase();
    const visible = conversations.filter((c) => !query || matches(c, query));
    if (!visible.length) {
      const empty = document.createElement('div');
      empty.className = 'empty-list';
      empty.textContent = query ? 'No conversation matches.' : 'No conversations yet.';
      list.appendChild(empty);
      return;
    }
    for (const conversation of visible) {
      const row = document.createElement('div');
      row.className = 'conversation' + (conversation.id === currentId ? ' active' : '');
      const title = document.createElement('span');
      title.className = 'conversation-title';
      title.textContent = conversation.title;
      const date = document.createElement('span');
      date.className = 'conversation-date';
      date.textContent = new Date(conversation.updatedAt).toLocaleDateString();
      const rename = document.createElement('button');
      rename.className = 'message-action';
      rename.textContent = '✎';
      rename.title = 'Rename';
      rename.onclick = (e) => { e.stopPropagation(); renameConversation(conversation.id); };
      const remove = document.createElement('button');
      remove.className = 'message-action';
      remove.textContent = '✕';
      remove.title = 'Delete';
      remove.onclick = (e) => { e.stopPropagation(); deleteConversation(conversation.id); };
      row.append(title, date, rename, remove);
      row.onclick = () => {
        currentId = conversation.id;
        renderSidebar();
        renderMessages();
        if (window.innerWidth <= 760) {
          document.getElementById('sidebar').classList.remove('open');
        }
      };
      list.appendChild(row);
    }
  }

  function matches(conversation, query) {
    if (conversation.title.toLowerCase().includes(query)) {
      return true;
    }
    return conversation.messages.some((m) => String(m.content || '').toLowerCase().includes(query));
  }

  // ---------- messages ----------

  function attachmentParts(message) {
    const images = (message.attachments || []).filter((a) => a.kind === 'image');
    const text = attachmentText(message);
    if (!images.length) {
      return text;
    }
    const parts = [{ type: 'text', text: text }];
    for (const image of images) {
      parts.push({ type: 'image_url', image_url: { url: image.data } });
    }
    return parts;
  }

  function requestMessages(conversation) {
    const out = [];
    const system = conversation.system || settings.defaultSystem;
    if (system && system.trim()) {
      out.push({ role: 'system', content: system.trim() });
    }
    for (const message of conversation.messages) {
      if (message.role === 'user') {
        out.push({ role: 'user', content: attachmentParts(message) });
      } else if (message.role === 'assistant') {
        out.push({ role: 'assistant', content: message.content });
      }
    }
    return out;
  }

  function requestBody(conversation) {
    const body = { model: selectedModel(), messages: requestMessages(conversation), stream: true };
    const number = (value) => (String(value).trim() === '' ? null : Number(value));
    const temperature = number(settings.temperature);
    const topP = number(settings.topP);
    const maxTokens = number(settings.maxTokens);
    const seed = number(settings.seed);
    if (temperature !== null) body.temperature = temperature;
    if (topP !== null) body.top_p = topP;
    if (maxTokens !== null && Number.isInteger(maxTokens) && maxTokens > 0) body.max_tokens = maxTokens;
    if (seed !== null && Number.isInteger(seed)) body.seed = seed;
    return body;
  }

  function renderMessages() {
    const container = document.getElementById('messages');
    container.innerHTML = '';
    const conversation = current();
    const welcome = document.getElementById('welcome');
    const empty = !conversation || !conversation.messages.length;
    welcome.hidden = !empty;
    welcome.querySelector('#welcome-note').textContent = serverProps && serverProps.model_path
      ? 'Chatting with ' + serverProps.model_path
      : 'A conversation with the model the server serves.';
    if (empty) {
      return;
    }
    for (let i = 0; i < conversation.messages.length; i++) {
      container.appendChild(messageElement(conversation, conversation.messages[i], i));
    }
    container.scrollTop = container.scrollHeight;
  }

  function messageElement(conversation, message, index) {
    const element = document.createElement('article');
    element.className = 'message ' + (message.error ? 'error' : message.role);
    element.appendChild(messageHead(conversation, message, index));
    const body = document.createElement('div');
    body.className = 'message-body';
    if (message.role === 'assistant' || message.error) {
      body.innerHTML = message.error ? md().escapeHtml(message.content) : md().render(message.content);
    } else {
      renderUserBody(body, message);
    }
    element.appendChild(body);
    return element;
  }

  function messageHead(conversation, message, index) {
    const head = document.createElement('div');
    head.className = 'message-head';
    const role = document.createElement('span');
    role.className = 'role';
    role.textContent = message.error ? 'error' : message.role === 'user' ? 'You' : selectedModel() || 'assistant';
    head.appendChild(role);
    head.appendChild(document.createElement('span')).className = 'spacer';
    if (message.role === 'assistant' && !message.error && !streamingMessageIs(message)) {
      head.appendChild(actionButton('Copy', () => copyText(message.content)));
      if (index === conversation.messages.length - 1) {
        head.appendChild(actionButton('Regenerate', () => regenerate()));
      }
    }
    if (message.role === 'user' && !stream) {
      head.appendChild(actionButton('Edit', () => editUserMessage(conversation, index)));
    }
    return head;
  }

  function actionButton(label, onClick) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'message-action';
    button.textContent = label;
    button.onclick = onClick;
    return button;
  }

  function streamingMessageIs(message) {
    return stream && stream.message === message;
  }

  function renderUserBody(body, message) {
    const images = (message.attachments || []).filter((a) => a.kind === 'image');
    const files = (message.attachments || []).filter((a) => a.kind === 'text');
    if (images.length || files.length) {
      const strip = document.createElement('div');
      strip.className = 'attachments-in-message';
      for (const image of images) {
        const img = document.createElement('img');
        img.src = image.data;
        img.alt = image.name;
        strip.appendChild(img);
      }
      for (const file of files) {
        const chip = document.createElement('span');
        chip.className = 'attachment-chip';
        chip.textContent = '📄 ' + file.name;
        strip.appendChild(chip);
      }
      body.appendChild(strip);
    }
    const text = document.createElement('div');
    text.innerHTML = md().render(message.content);
    body.appendChild(text);
  }

  function copyText(text) {
    navigator.clipboard.writeText(String(text)).then(
      () => {}, () => showError('the clipboard refuses the copy')
    );
  }

  function editUserMessage(conversation, index) {
    const message = conversation.messages[index];
    if (!message) {
      return;
    }
    conversation.messages = conversation.messages.slice(0, index);
    attachments = (message.attachments || []).slice();
    document.getElementById('input').value = message.content;
    saveConversations();
    renderAttachments();
    renderMessages();
    document.getElementById('input').focus();
  }

  function regenerate() {
    const conversation = current();
    if (!conversation || stream) {
      return;
    }
    while (conversation.messages.length && conversation.messages[conversation.messages.length - 1].role !== 'user') {
      conversation.messages.pop();
    }
    saveConversations();
    renderMessages();
    if (conversation.messages.length) {
      sendCompletion(conversation);
    }
  }

  // ---------- streaming ----------

  function send() {
    const input = document.getElementById('input');
    const text = input.value.trim();
    if (!text && !attachments.length) {
      return;
    }
    if (stream) {
      showError('a response is still streaming; stop it first');
      return;
    }
    let conversation = current();
    if (!conversation) {
      conversation = createConversation(false);
    }
    const message = {
      role: 'user',
      content: text || (attachments.length ? 'Please read the attached file.' : ''),
      attachments: attachments.slice()
    };
    conversation.messages.push(message);
    autoTitle(conversation);
    conversation.updatedAt = Date.now();
    input.value = '';
    attachments = [];
    renderAttachments();
    saveConversations();
    renderSidebar();
    renderMessages();
    sendCompletion(conversation);
  }

  async function sendCompletion(conversation) {
    const assistant = { role: 'assistant', content: '', reasoning: '' };
    conversation.messages.push(assistant);
    renderMessages();
    const bodyEl = document.querySelector('#messages .message:last-child .message-body');
    const element = document.querySelector('#messages .message:last-child');
    element.classList.add('streaming');
    bodyEl.classList.add('streaming');
    const controller = new AbortController();
    stream = { controller, message: assistant, element, bodyEl, timer: null, dirty: false };
    document.getElementById('send').hidden = true;
    document.getElementById('stop').hidden = false;
    try {
      const response = await fetch(BASE + '/v1/chat/completions', {
        method: 'POST',
        headers: headers(true),
        body: JSON.stringify(requestBody(conversation)),
        signal: controller.signal
      });
      if (!response.ok) {
        throw await errorOf(response);
      }
      await readStream(response, assistant, conversation);
    } catch (error) {
      finishStream(conversation, assistant);
      if (error.name === 'AbortError') {
        assistant.content += (assistant.content ? '\n\n' : '') + '*(stopped)*';
        conversation.messages.pop();
      } else if (assistant.content) {
        // a partial answer stays in the chat; the failure is named above the composer
        showError(error.message || 'the request failed');
      } else {
        conversation.messages.pop();
        showError(error.message || 'the request failed');
      }
    }
    stream = null;
    document.getElementById('send').hidden = false;
    document.getElementById('stop').hidden = true;
    saveConversations();
    renderMessages();
  }

  async function readStream(response, assistant, conversation) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const read = await reader.read();
      if (read.done) {
        break;
      }
      buffer += decoder.decode(read.value, { stream: true });
      let split;
      while ((split = buffer.indexOf('\n\n')) >= 0) {
        handleFrame(buffer.slice(0, split), assistant, conversation);
        buffer = buffer.slice(split + 2);
      }
    }
    if (buffer.trim()) {
      handleFrame(buffer, assistant, conversation);
    }
    finishStream(conversation, assistant);
  }

  function handleFrame(frame, assistant, conversation) {
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) {
        continue;
      }
      const payload = line.slice(5).trim();
      if (!payload) {
        continue;
      }
      if (payload === '[DONE]') {
        continue;
      }
      let chunk;
      try {
        chunk = JSON.parse(payload);
      } catch (e) {
        continue;
      }
      if (chunk.error) {
        throw new Error(chunk.error.message || 'the stream failed');
      }
      const choice = chunk.choices && chunk.choices[0];
      if (choice && choice.delta) {
        if (choice.delta.reasoning_content) {
          assistant.reasoning += choice.delta.reasoning_content;
        }
        if (choice.delta.content) {
          assistant.content += choice.delta.content;
        }
        scheduleStreamRender();
      }
      if (choice && choice.finish_reason) {
        assistant.finishReason = choice.finish_reason;
      }
    }
  }

  function scheduleStreamRender() {
    const active = stream;
    if (!active || active.dirty) {
      return;
    }
    active.dirty = true;
    if (active.timer === null) {
      active.timer = setTimeout(() => {
        if (stream !== active) {
          return;
        }
        active.timer = null;
        active.dirty = false;
        renderStreamBody(active);
        const container = document.getElementById('messages');
        container.scrollTop = container.scrollHeight;
      }, 80);
    }
  }

  function renderStreamBody(active) {
    active.bodyEl.innerHTML = streamHtml(active.message);
    const container = document.getElementById('messages');
    container.scrollTop = container.scrollHeight;
  }

  function streamHtml(message) {
    let html = '';
    if (message.reasoning) {
      html += '<details class="reasoning"><summary>Thinking</summary><div>' +
        md().render(message.reasoning) + '</div></details>';
    }
    html += md().render(message.content);
    return html;
  }

  function finishStream(conversation, assistant) {
    const active = stream;
    if (active && active.timer !== null) {
      clearTimeout(active.timer);
      active.timer = null;
    }
    assistant.updatedAt = Date.now();
    conversation.updatedAt = Date.now();
    stopSlotsPolling();
  }

  function stopStreaming() {
    if (stream) {
      stream.controller.abort();
    }
  }

  // ---------- attachments ----------

  function renderAttachments() {
    const strip = document.getElementById('attachments');
    strip.innerHTML = '';
    for (let i = 0; i < attachments.length; i++) {
      const item = attachments[i];
      const chip = document.createElement('span');
      chip.className = 'attachment-chip';
      chip.textContent = (item.kind === 'image' ? '🖼 ' : '📄 ') + item.name;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'message-action';
      remove.textContent = '✕';
      remove.title = 'Remove';
      remove.onclick = () => {
        attachments.splice(i, 1);
        renderAttachments();
      };
      chip.appendChild(remove);
      strip.appendChild(chip);
    }
  }

  function addFiles(files) {
    for (const file of files) {
      const extension = (file.name.split('.').pop() || '').toLowerCase();
      if (IMAGE_TYPES.includes(file.type) && IMAGE_EXTENSIONS.includes(extension)) {
        if (file.size > IMAGE_LIMIT) {
          showError(file.name + ' is over the 4 MB image limit');
          continue;
        }
        const reader = new FileReader();
        reader.onload = () => {
          attachments.push({ kind: 'image', name: file.name, mime: file.type, data: String(reader.result) });
          renderAttachments();
        };
        reader.onerror = () => showError(file.name + ' does not read');
        reader.readAsDataURL(file);
      } else if (file.type.startsWith('text/') || TEXT_EXTENSIONS.includes(extension)) {
        if (file.size > TEXT_LIMIT) {
          showError(file.name + ' is over the 200 KB text limit');
          continue;
        }
        const reader = new FileReader();
        reader.onload = () => {
          attachments.push({ kind: 'text', name: file.name, mime: file.type || 'text/plain', data: String(reader.result) });
          renderAttachments();
        };
        reader.onerror = () => showError(file.name + ' does not read');
        reader.readAsText(file);
      } else {
        showError('unsupported file type: ' + file.name);
      }
    }
  }

  function attachmentText(message) {
    const files = (message.attachments || []).filter((a) => a.kind === 'text');
    if (!files.length) {
      return message.content;
    }
    let out = message.content;
    for (const file of files) {
      out += '\n\n**' + file.name + '**\n\n```\n' + file.data + '\n```';
    }
    return out;
  }

  // ---------- panels ----------

  function kvTable(pairs) {
    const table = document.createElement('table');
    for (const [key, value] of pairs) {
      const row = document.createElement('tr');
      const th = document.createElement('th');
      th.textContent = key;
      const td = document.createElement('td');
      td.textContent = value;
      row.append(th, td);
      table.appendChild(row);
    }
    return table;
  }

  function renderServerInfo() {
    const target = document.getElementById('server-info');
    target.innerHTML = '';
    if (!serverProps) {
      const hint = document.createElement('p');
      hint.className = 'hint';
      hint.textContent = propsError ? propsError.message : 'The server does not answer /props.';
      target.appendChild(hint);
      return;
    }
    if (serverProps.role === 'router') {
      target.appendChild(kvTable([
        ['role', serverProps.role],
        ['max instances', serverProps.max_instances == null ? '' : String(serverProps.max_instances)],
        ['models autoload', String(serverProps.models_autoload === true)],
        ['model path', String(serverProps.model_path || '')]
      ]));
      const hint = document.createElement('p');
      hint.className = 'hint';
      hint.textContent = 'Select a model to see its serving properties.';
      target.appendChild(hint);
      return;
    }
    const generation = serverProps.default_generation_settings || {};
    const media = serverProps.modalities || {};
    target.appendChild(kvTable([
      ['model', serverProps.model_path || ''],
      ['alias', serverProps.model_alias || ''],
      ['architecture', serverProps.model_arch || ''],
      ['context', generation.n_ctx == null ? '' : String(generation.n_ctx)],
      ['slots', serverProps.total_slots == null ? '' : String(serverProps.total_slots)],
      ['vision', String(media.vision === true)],
      ['audio', String(media.audio === true)],
      ['sleeping', String(serverProps.is_sleeping === true)],
      ['chat template', String(serverProps.chat_template || '').slice(0, 200)]
    ]));
  }

  async function renderSlots() {
    const target = document.getElementById('slots-info');
    target.innerHTML = '';
    const model = selectedModel();
    const path = model
      ? '/slots?model=' + encodeURIComponent(model) + (routerMode() ? '&autoload=false' : '')
      : '/slots';
    try {
      const slots = await api(path);
      if (!Array.isArray(slots) || !slots.length) {
        target.innerHTML = '<p class="hint">The server reports no slots.</p>';
        return;
      }
      const table = document.createElement('table');
      const head = document.createElement('tr');
      for (const key of Object.keys(slots[0])) {
        const th = document.createElement('th');
        th.textContent = key;
        head.appendChild(th);
      }
      table.appendChild(head);
      for (const slot of slots) {
        const row = document.createElement('tr');
        for (const key of Object.keys(slots[0])) {
          const td = document.createElement('td');
          td.textContent = String(slot[key]);
          row.appendChild(td);
        }
        table.appendChild(row);
      }
      target.appendChild(table);
    } catch (error) {
      const hint = document.createElement('p');
      hint.className = 'hint';
      hint.textContent = routerMode() && model ? 'Slots for ' + model + ': ' + error.message : error.message;
      target.appendChild(hint);
    }
  }

  function startSlotsPolling() {
    stopSlotsPolling();
    slotsTimer = setInterval(() => {
      if (document.getElementById('slots-dialog').open) {
        renderSlots();
      }
    }, 1500);
  }

  function stopSlotsPolling() {
    if (slotsTimer !== null) {
      clearInterval(slotsTimer);
      slotsTimer = null;
    }
  }

  // ---------- import / export ----------

  function exportConversations() {
    const payload = { version: 1, exportedAt: new Date().toISOString(), conversations: conversations };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'neuralfree-conversations.json';
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function importConversations(file) {
    const reader = new FileReader();
    reader.onload = () => {
      let parsed;
      try {
        parsed = JSON.parse(String(reader.result));
      } catch (e) {
        showError('the file is not valid JSON: ' + e.message);
        return;
      }
      const list = parsed && parsed.conversations;
      if (!Array.isArray(list)) {
        showError('invalid conversation file: "conversations" is missing');
        return;
      }
      for (const item of list) {
        if (!item || !Array.isArray(item.messages) ||
            item.messages.some((m) => !m || typeof m !== 'object' || typeof m.role !== 'string')) {
          showError('invalid conversation file: a conversation has no message list');
          return;
        }
      }
      const known = new Set(conversations.map((c) => c.id));
      for (const item of list) {
        const conversation = {
          id: known.has(item.id) ? newId() : String(item.id || newId()),
          title: String(item.title || 'Imported chat'),
          createdAt: Number(item.createdAt) || Date.now(),
          updatedAt: Number(item.updatedAt) || Date.now(),
          system: item.system || '',
          messages: item.messages.map((m) => ({
            role: m.role === 'assistant' ? 'assistant' : m.role === 'system' ? 'system' : 'user',
            content: String(m.content == null ? '' : m.content),
            reasoning: m.reasoning ? String(m.reasoning) : ''
          }))
        };
        known.add(conversation.id);
        conversations.unshift(conversation);
      }
      currentId = conversations[0].id;
      saveConversations();
      renderSidebar();
      renderMessages();
    };
    reader.onerror = () => showError('the file does not read');
    reader.readAsText(file);
  }

  // ---------- settings ----------

  function openSettings() {
    document.getElementById('setting-api-key').value = settings.apiKey;
    document.getElementById('setting-theme').value = settings.theme;
    document.getElementById('setting-system').value = settings.defaultSystem;
    document.getElementById('setting-temperature').value = settings.temperature;
    document.getElementById('setting-top-p').value = settings.topP;
    document.getElementById('setting-max-tokens').value = settings.maxTokens;
    document.getElementById('setting-seed').value = settings.seed;
    document.getElementById('settings-dialog').showModal();
  }

  function saveSettingsForm() {
    settings.apiKey = document.getElementById('setting-api-key').value.trim();
    settings.theme = document.getElementById('setting-theme').value;
    settings.defaultSystem = document.getElementById('setting-system').value;
    settings.temperature = document.getElementById('setting-temperature').value.trim();
    settings.topP = document.getElementById('setting-top-p').value.trim();
    settings.maxTokens = document.getElementById('setting-max-tokens').value.trim();
    settings.seed = document.getElementById('setting-seed').value.trim();
    saveSettings();
    applyTheme();
  }

  function applyTheme() {
    const dark = settings.theme === 'dark' ||
      (settings.theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  }

  // ---------- errors ----------

  let errorTimer = null;

  function showError(message) {
    const box = document.getElementById('composer-error');
    box.textContent = message;
    box.hidden = false;
    if (errorTimer !== null) {
      clearTimeout(errorTimer);
    }
    errorTimer = setTimeout(() => {
      box.hidden = true;
      errorTimer = null;
    }, 8000);
  }

  // ---------- boot ----------

  function autoSize() {
    const input = document.getElementById('input');
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
  }

  function init() {
    settings = Object.assign(defaultSettings(), loadJson(SETTINGS_KEY, {}));
    conversations = loadJson(CONVERSATIONS_KEY, []);
    if (!Array.isArray(conversations)) {
      conversations = [];
    }
    applyTheme();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);

    document.getElementById('new-chat').onclick = () => createConversation(true);
    document.getElementById('search').oninput = renderSidebar;
    document.getElementById('settings-btn').onclick = openSettings;
    document.getElementById('settings-save').onclick = saveSettingsForm;
    document.getElementById('server-info-btn').onclick = () => {
      renderServerInfo();
      document.getElementById('server-dialog').showModal();
    };
    document.getElementById('server-refresh').onclick = async () => {
      await refreshServer();
      renderServerInfo();
    };
    document.getElementById('slots-btn').onclick = () => {
      renderSlots();
      startSlotsPolling();
      document.getElementById('slots-dialog').showModal();
    };
    document.getElementById('slots-dialog').addEventListener('close', stopSlotsPolling);
    document.getElementById('slots-refresh').onclick = renderSlots;
    document.getElementById('export-btn').onclick = exportConversations;
    document.getElementById('import-btn').onclick = () => document.getElementById('import-file').click();
    document.getElementById('import-file').onchange = (e) => {
      if (e.target.files.length) {
        importConversations(e.target.files[0]);
      }
      e.target.value = '';
    };
    document.getElementById('model-select').onchange = async (e) => {
      settings.model = e.target.value;
      saveSettings();
      renderMessages();
      await refreshProps();
      updateModelLoad();
      if (document.getElementById('server-dialog').open) {
        renderServerInfo();
      }
      if (document.getElementById('slots-dialog').open) {
        renderSlots();
      }
    };
    document.getElementById('model-load').onclick = loadSelectedModel;
    document.getElementById('attach-btn').onclick = () => document.getElementById('file-input').click();
    document.getElementById('file-input').onchange = (e) => {
      addFiles(Array.from(e.target.files));
      e.target.value = '';
    };
    document.getElementById('send').onclick = send;
    document.getElementById('stop').onclick = stopStreaming;
    document.getElementById('toggle-sidebar').onclick = () =>
      document.getElementById('sidebar').classList.toggle('open');

    const input = document.getElementById('input');
    input.addEventListener('input', autoSize);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        send();
      }
    });
    const composer = document.querySelector('.composer');
    composer.addEventListener('dragover', (e) => {
      e.preventDefault();
      composer.classList.add('dragover');
    });
    composer.addEventListener('dragleave', () => composer.classList.remove('dragover'));
    composer.addEventListener('drop', (e) => {
      e.preventDefault();
      composer.classList.remove('dragover');
      addFiles(Array.from(e.dataTransfer.files));
    });
    input.addEventListener('paste', (e) => {
      const files = Array.from(e.clipboardData.files || []);
      if (files.length) {
        e.preventDefault();
        addFiles(files);
      }
    });

    document.getElementById('messages').addEventListener('click', (e) => {
      const copy = e.target.closest('.copy-code');
      if (copy) {
        const code = copy.closest('.code-block').querySelector('code');
        navigator.clipboard.writeText(code.textContent).then(() => {
          copy.textContent = 'Copied';
          setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
        }, () => showError('the clipboard refuses the copy'));
      }
    });

    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        createConversation(true);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        document.getElementById('sidebar').classList.toggle('open');
      }
    });

    if (!conversations.length) {
      createConversation(false);
    }
    currentId = currentId || conversations[0].id;
    renderSidebar();
    renderMessages();
    refreshServer();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { init: init };
})();
