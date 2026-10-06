(() => {
  const data = window.DEMO_DATA;
  const groups = [
    ["all", "All systems"], ["low", "Low rate"], ["main", "Main rate"], ["spectral", "Spectral controls"]
  ];
  let activeGroup = "main";
  let activeCut = data.cuts[0].cut_id;
  const tabs = document.getElementById("group-tabs");
  const cutSelect = document.getElementById("cut-select");
  const list = document.getElementById("system-list");

  const fmt = (value, digits) => value == null ? "--" : Number(value).toFixed(digits);
  const el = (tag, className, text) => {
    const node = document.createElement(tag); if (className) node.className = className;
    if (text != null) node.textContent = text; return node;
  };
  groups.forEach(([key, label]) => {
    const button = el("button", "", label); button.type = "button"; button.role = "tab";
    button.setAttribute("aria-selected", String(key === activeGroup));
    button.addEventListener("click", () => { activeGroup = key; render(); }); tabs.appendChild(button);
  });
  data.cuts.forEach((cut, index) => {
    const option = document.createElement("option"); option.value = cut.cut_id;
    option.textContent = `${index + 1}. ${cut.cut_id}`; cutSelect.appendChild(option);
  });
  cutSelect.addEventListener("change", () => { activeCut = cutSelect.value; render(); });

  function render() {
    document.querySelectorAll("[role=tab]").forEach((button, index) =>
      button.setAttribute("aria-selected", String(groups[index][0] === activeGroup)));
    const cut = data.cuts.find((item) => item.cut_id === activeCut);
    document.getElementById("cut-id").textContent = cut.cut_id;
    document.getElementById("cut-meta").textContent = `Speaker ${cut.speaker_id} · ${cut.duration.toFixed(2)} s`;
    document.getElementById("transcript").textContent = cut.transcript;
    list.replaceChildren();
    data.systems.filter((system) => system.groups.includes(activeGroup)).forEach((system) => {
      const row = el("article", "system-row"); row.dataset.family = system.family;
      const name = el("div", "system-name"); name.append(el("strong", "", system.label));
      if (system.key === "reference") name.append(el("span", "", "Evaluator ceiling"));
      row.append(name);
      const values = [
        ["Rate", system.nominal_bitrate_kbps == null ? "--" : `${fmt(system.nominal_bitrate_kbps, 3)}k`, "metric rate"],
        ["WER", `${fmt(system.wer_percent, 2)}%`, "metric"], ["ViSQOL", fmt(system.visqol, 3), "metric"],
        ["PESQ", fmt(system.pesq, 3), "metric"], ["Spk", fmt(system.speaker_cosine, 3), "metric"]
      ];
      values.forEach(([label, value, className]) => { const cell = el("span", className, value);
        cell.dataset.label = label; row.append(cell); });
      const player = el("div", "player");
      const audio = el("audio"); audio.controls = true; audio.preload = "metadata"; audio.src = system.audio[activeCut].file;
      audio.addEventListener("play", () => { document.querySelectorAll("audio").forEach((other) => { if (other !== audio) other.pause(); });
        document.querySelectorAll(".system-row").forEach((other) => other.classList.remove("playing")); row.classList.add("playing"); });
      audio.addEventListener("pause", () => row.classList.remove("playing")); player.append(audio); row.append(player); list.append(row);
    });
  }
  render();
})();
