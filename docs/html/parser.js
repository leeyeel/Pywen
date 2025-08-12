/**
 * Parse trajectory-style JSON into a unified structure for the visualizer.
 *
 * Export: window.parseConversationLog(rawJsonString)
 */
function parseConversationLog(rawJsonString) {
  const ts = () => new Date().toISOString();

  function cloneJSON(x) {
    try { return JSON.parse(JSON.stringify(x)); } catch { return x; }
  }

  // ---- Parse & validate ----
  let t;
  try {
    t = JSON.parse(rawJsonString);
  } catch (e) {
    throw new Error('File is not valid JSON');
  }
  if (!t || typeof t !== 'object' || !Array.isArray(t.llm_interactions)) {
    throw new Error('Not a supported trajectory JSON: missing llm_interactions');
  }

  // ---- Build tool catalog from tools_available union ----
  const tool_defs = {}; // name -> def
  const allNames = new Set();
  for (const it of t.llm_interactions || []) {
    (it.tools_available || []).forEach(n => allNames.add(String(n)));
  }
  for (const name of allNames) {
    tool_defs[name] = {
      name,
      type: 'tool',
      description: '',
      input_schema: { type: 'object' }
    };
  }

  // ---- Build conversations (one per interaction) ----
  const conversations = (t.llm_interactions || [])
    .slice()
    .sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp))
    .map((it, idx) => {
      const sysBlocks = [];
      const msgs = [];
      for (const m of (it.input_messages || [])) {
        if (m.role === 'system') sysBlocks.push({ type: 'text', text: String(m.content ?? '') });
        msgs.push({ role: m.role, content: String(m.content ?? '') });
      }

      const input = {
        provider: it.provider || t.provider || '',
        model: it.model || t.model || '',
        system: sysBlocks,
        messages: msgs,
        tools: Array.isArray(it.tools_available) ? it.tools_available.slice() : []
      };

      const resp = it.response || {};
      const data = {
        id: `traj_${idx}`,
        role: 'assistant',
        model: resp.model || it.model || t.model || '',
        text: typeof resp.content === 'string' ? resp.content : '',
        usage: cloneJSON(resp.usage || {}),
        stop_reason: resp.finish_reason || null,
        tools: Array.isArray(resp.tool_calls)
          ? resp.tool_calls.map((c, i) => ({ id: `call_${i}` , name: c.name || 'tool', preview: JSON.stringify(c, null, 2), durationMs: null }))
          : []
      };

      return {
        uid: `traj_${idx}`,
        started_at: it.timestamp || t.start_time || null,
        finished_at: it.timestamp || null,
        request_id: null,
        input,
        context: {
          current_task: it.current_task || null
        },
        result: { type: 'output', data }
      };
    });

  return {
    session_title: t.task ? `Trajectory: ${t.task}` : (t.model ? `Trajectory (${t.model})` : 'Trajectory'),
    tool_defs,
    prompts: {},
    prompt_counts: {},
    prompt_kind_guess: {},
    conversations,
    meta: {
      start_time: t.start_time || null,
      end_time: t.end_time || null,
      provider: t.provider || null,
      model: t.model || null,
      ax_steps: t.max_steps ?? null,
      success: !!t.success,
      execution_time: t.execution_time ?? null,
      total_tokens: t.total_tokens ?? null,
      total_input_tokens: t.total_input_tokens ?? null,
      total_output_tokens: t.total_output_tokens ?? null,
      task: t.task || null,
      context_tokens: t.context_tokens ?? null,
      agent_steps_count: Array.isArray(t.agent_steps) ? t.agent_steps.length : 0
    },
    raw: t,
    generated_at: ts()
  };
}

if (typeof module !== 'undefined' && module.exports) module.exports = parseConversationLog;
if (typeof window !== 'undefined') window.parseConversationLog = parseConversationLog;

