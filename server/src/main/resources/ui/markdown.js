/*
 * NeuralFree chat UI: the message markdown renderer.
 *
 * A GFM subset without dependencies: headings, paragraphs, fenced code blocks with a copy
 * button, inline code, bold, italic, strikethrough, links and images (safe schemes only),
 * autolinks, block quotes, unordered and ordered lists with nesting, horizontal rules and
 * tables with alignment. HTML in the source is always escaped, never interpreted.
 */
window.NfMarkdown = (function () {
  'use strict';

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * A scheme a link or image may point at; anything else renders as plain text. A root-relative path must not
   * start with a second slash or a backslash: browsers resolve both as a link to another host.
   */
  function safeUrl(url) {
    return /^(https?:\/\/|mailto:|\/(?![\/\\])|#)/i.test(url) ? url : null;
  }

  /**
   * A link destination: runs of plain characters (no spaces or parentheses) and balanced paren groups, so a URL
   * such as `.../Python_(programming_language)` renders whole; the first unbalanced `)` ends it (three nesting
   * levels cover real URLs).
   */
  const DESTINATION = '(?:[^()\\s]|\\((?:[^()\\s]|\\((?:[^()\\s]|\\([^()\\s]*\\))*\\))*\\))+';

  const INLINE_RULES = [
    { re: /`([^`]+)`/, emit: (m) => '<code class="inline">' + escapeHtml(m[1]) + '</code>' },
    { re: /\*\*\*([^*]+(?:\*[^*]+)*)\*\*\*/, emit: (m) => '<strong><em>' + renderInline(m[1]) + '</em></strong>' },
    { re: /\*\*([^*]+(?:\*[^*]+)*)\*\*/, emit: (m) => '<strong>' + renderInline(m[1]) + '</strong>' },
    { re: /(^|[^\w*])\*([^*\s][^*]*)\*/, emit: (m) => m[1] + '<em>' + renderInline(m[2]) + '</em>' },
    { re: /__([^_]+(?:_[^_]+)*)__/, emit: (m) => '<strong>' + renderInline(m[1]) + '</strong>' },
    { re: /(^|[^\w_])_([^_\s][^_]*)_(?![A-Za-z0-9])/, emit: (m) => m[1] + '<em>' + renderInline(m[2]) + '</em>' },
    { re: /~~([^~]+)~~/, emit: (m) => '<del>' + renderInline(m[1]) + '</del>' },
    {
      re: new RegExp('!\\[([^\\]]*)\\]\\(\\s*(' + DESTINATION + ')(?:\\s+"([^"]*)")?\\s*\\)'),
      emit: (m) => {
        const url = safeUrl(m[2]);
        return url
          ? '<img src="' + escapeHtml(url) + '" alt="' + escapeHtml(m[1]) + '" title="' + escapeHtml(m[3] || '') + '">'
          : escapeHtml(m[1]);
      }
    },
    {
      re: new RegExp('\\[([^\\]]*)\\]\\(\\s*(' + DESTINATION + ')(?:\\s+"([^"]*)")?\\s*\\)'),
      emit: (m) => {
        const url = safeUrl(m[2]);
        const label = renderInline(m[1] || m[2]);
        return url
          ? '<a href="' + escapeHtml(url) + '" title="' + escapeHtml(m[3] || '') +
            '" target="_blank" rel="noopener noreferrer">' + label + '</a>'
          : label;
      }
    },
    {
      re: /(^|[\s(])(https?:\/\/[^\s<>()[\]]+[^\s<>()[\].,;:!?'"])/,
      emit: (m) => m[1] + '<a href="' + escapeHtml(m[2]) + '" target="_blank" rel="noopener noreferrer">' +
        escapeHtml(m[2]) + '</a>'
    }
  ];

  /** Renders one inline run: the earliest rule match wins, the skipped text is escaped. */
  function renderInline(text) {
    let out = '';
    let rest = String(text);
    while (rest.length) {
      let best = null;
      for (const rule of INLINE_RULES) {
        const match = rule.re.exec(rest);
        if (match && (best === null || match.index < best.index)) {
          best = { index: match.index, match, rule };
        }
      }
      if (best === null) {
        break;
      }
      out += escapeHtml(rest.slice(0, best.index));
      out += best.rule.emit(best.match);
      rest = rest.slice(best.index + best.match[0].length);
    }
    return out + escapeHtml(rest);
  }

  function codeBlockHtml(lang, code) {
    return '<div class="code-block"><div class="code-head"><span>' + escapeHtml(lang || 'text') +
      '</span><button type="button" class="message-action copy-code">Copy</button></div>' +
      '<pre><code>' + escapeHtml(code) + '</code></pre></div>';
  }

  function fenceOf(line) {
    const match = /^ {0,3}(`{3,}|~{3,})\s*(\S*)\s*$/.exec(line);
    return match ? { marker: match[1][0], length: match[1].length, lang: match[2] } : null;
  }

  /** A line closes a fence when it uses the same marker and is not shorter. */
  function closesFence(line, fence) {
    const close = fenceOf(line);
    return close !== null && close.marker === fence.marker && close.length >= fence.length;
  }

  function headingHtml(line) {
    const match = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
    if (!match) {
      return null;
    }
    const level = match[1].length;
    return '<h' + level + '>' + renderInline(match[2]) + '</h' + level + '>';
  }

  function isRule(line) {
    return /^ {0,3}((-\s*){3,}|(\*\s*){3,}|(_\s*){3,})$/.test(line);
  }

  function isTableRow(line) {
    const text = line.trim();
    return text.startsWith('|') || (text.includes('|') && /\S/.test(text));
  }

  function splitCells(line) {
    let text = line.trim().replace(/^\|/, '').replace(/\|$/, '');
    return text.split('|').map((cell) => cell.trim());
  }

  function tableAlignments(separator) {
    return splitCells(separator).map((cell) => {
      const left = cell.startsWith(':');
      const right = cell.endsWith(':');
      if (left && right) return 'text-align:center;';
      if (right) return 'text-align:right;';
      if (left) return 'text-align:left;';
      return '';
    });
  }

  function tableHtml(head, alignments, rows) {
    let out = '<table><thead><tr>';
    head.forEach((cell, i) => {
      out += '<th style="' + alignments[i] + '">' + renderInline(cell) + '</th>';
    });
    out += '</tr></thead><tbody>';
    for (const row of rows) {
      out += '<tr>';
      for (let i = 0; i < head.length; i++) {
        out += '<td style="' + alignments[i] + '">' + renderInline(row[i] || '') + '</td>';
      }
      out += '</tr>';
    }
    return out + '</tbody></table>';
  }

  /** Collects the consecutive lines of one list; a lazy continuation joins the previous item. */
  function listItems(lines, start, baseIndent) {
    const items = [];
    let i = start;
    while (i < lines.length) {
      const match = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/.exec(lines[i]);
      if (match) {
        const indent = match[1].length;
        if (items.length && indent < baseIndent) {
          break;
        }
        items.push({ indent: Math.min(indent, baseIndent + 2), ordered: /\d/.test(match[2]), text: match[3] });
        i++;
      } else if (!lines[i].trim()) {
        let next = i + 1;
        while (next < lines.length && !lines[next].trim()) next++;
        if (next < lines.length && /^(\s*)([-*+]|\d{1,9}[.)])\s+/.test(lines[next])) {
          i = next;
        } else {
          break;
        }
      } else if (items.length && /^\s{2,}\S/.test(lines[i])) {
        items[items.length - 1].text += ' ' + lines[i].trim();
        i++;
      } else {
        break;
      }
    }
    return { items, next: i };
  }

  function listHtml(items) {
    let pos = 0;
    let out = '';
    // consecutive items of one type and indent form a list; a switch of type or indent opens another
    while (pos < items.length) {
      out += level(items[pos].indent);
    }
    return out;

    function level(indent) {
      const ordered = items[pos].ordered;
      let html = ordered ? '<ol>' : '<ul>';
      while (pos < items.length && items[pos].indent >= indent && items[pos].ordered === ordered) {
        pos++;
        let inner = renderInline(items[pos - 1].text);
        while (pos < items.length && items[pos].indent > indent) {
          inner += level(items[pos].indent);
        }
        html += '<li>' + inner + '</li>';
      }
      return html + (ordered ? '</ol>' : '</ul>');
    }
  }

  function renderBlocks(lines) {
    let out = '';
    let paragraph = [];
    let i = 0;

    const flushParagraph = () => {
      if (paragraph.length) {
        out += '<p>' + paragraph.map(renderInline).join('<br>') + '</p>';
        paragraph = [];
      }
    };

    while (i < lines.length) {
      const line = lines[i];
      const fence = fenceOf(line);
      if (fence) {
        flushParagraph();
        const content = [];
        i++;
        while (i < lines.length && !closesFence(lines[i], fence)) {
          content.push(lines[i]);
          i++;
        }
        if (i < lines.length) {
          i++;
        }
        out += codeBlockHtml(fence.lang, content.join('\n'));
        continue;
      }
      const heading = headingHtml(line);
      if (heading) {
        flushParagraph();
        out += heading;
        i++;
        continue;
      }
      if (isRule(line)) {
        flushParagraph();
        out += '<hr>';
        i++;
        continue;
      }
      if (/^ {0,3}>/.test(line)) {
        flushParagraph();
        const quoted = [];
        while (i < lines.length && /^ {0,3}>/.test(lines[i])) {
          quoted.push(lines[i].replace(/^ {0,3}> ?/, ''));
          i++;
        }
        out += '<blockquote>' + renderBlocks(quoted) + '</blockquote>';
        continue;
      }
      const listMatch = /^(\s*)([-*+]|\d{1,9}[.)])\s+/.exec(line);
      if (listMatch) {
        flushParagraph();
        const collected = listItems(lines, i, listMatch[1].length);
        out += listHtml(collected.items);
        i = collected.next;
        continue;
      }
      if (isTableRow(line) && i + 1 < lines.length && /^ *\|? *:?-+:? *(\| *:?-+:? *)*\|? *$/.test(lines[i + 1])) {
        flushParagraph();
        const head = splitCells(line);
        const alignments = tableAlignments(lines[i + 1]);
        const rows = [];
        i += 2;
        while (i < lines.length && isTableRow(lines[i]) && lines[i].trim()) {
          rows.push(splitCells(lines[i]));
          i++;
        }
        out += tableHtml(head, alignments, rows);
        continue;
      }
      if (!line.trim()) {
        flushParagraph();
        i++;
        continue;
      }
      paragraph.push(line);
      i++;
    }
    flushParagraph();
    return out;
  }

  /** The HTML of one markdown text. */
  function render(text) {
    const normalized = String(text == null ? '' : text).replace(/\r\n?/g, '\n').replace(/\t/g, '    ');
    return renderBlocks(normalized.split('\n'));
  }

  return { render: render, escapeHtml: escapeHtml };
})();
