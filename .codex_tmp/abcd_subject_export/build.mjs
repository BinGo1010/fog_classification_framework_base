import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = "E:/fog-merged";
const sourceDir = path.join(
  repoRoot,
  "outputs/daphnet_residual_calibration_ABCD_3seed_seed20260807",
);
const outputDir = path.join(
  repoRoot,
  "outputs/019fbd53-c773-7cc3-9e6a-5e4ab2c4d00e",
);
const previewDir = path.join(outputDir, "previews");
const outputXlsx = path.join(outputDir, "daphnet_ABCD_subject_main_metrics.xlsx");
const outputMarkdown = path.join(outputDir, "daphnet_ABCD_subject_main_metrics.md");

function parseCsv(text) {
  text = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const [headers, ...body] = rows;
  return body.map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
  );
}

const subjectCsv = await fs.readFile(
  path.join(sourceDir, "subject_metrics_3seed_mean_std.csv"),
  "utf8",
);
const overallCsv = await fs.readFile(
  path.join(sourceDir, "group_summary_3seed_mean_std.csv"),
  "utf8",
);
const subjectRows = parseCsv(subjectCsv);
const overallRows = parseCsv(overallCsv);
const groups = ["A", "B", "C", "D"];
const subjects = ["S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09"];
const metricOrder = [
  ["accuracy", "Accuracy"],
  ["balanced_accuracy", "Balanced Accuracy"],
  ["precision", "FoG Precision"],
  ["sensitivity", "FoG Recall"],
  ["specificity", "Specificity"],
  ["f1", "FoG F1"],
  ["auprc", "PR-AUC"],
  ["auroc", "AUROC"],
];
const primaryMetrics = [
  { key: "accuracy", label: "Accuracy", meanColumn: "C", stdColumn: "D" },
  { key: "sensitivity", label: "FoG Recall", meanColumn: "I", stdColumn: "J" },
  { key: "specificity", label: "Specificity", meanColumn: "K", stdColumn: "L" },
  { key: "auprc", label: "PR-AUC", meanColumn: "O", stdColumn: "P" },
];
const methodDescriptions = {
  A: "Location–Scale Calibration；不做残差逐窗口中心化",
  B: "Location–Scale Calibration；Clip后逐窗口逐轴中心化",
  C: "仅Scale Calibration；Clip后逐窗口逐轴中心化",
  D: "不使用b和sigma；原始重构误差逐窗口逐轴中心化；不Clip",
};

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README");
const overall = workbook.worksheets.add("Overall");
const compare = workbook.worksheets.add("Subject Compare");
const raw = workbook.worksheets.add("Subject Data");

const navy = "#15324B";
const teal = "#147D7E";
const lightBlue = "#DCEAF4";
const lightTeal = "#DDF2EF";
const lightGray = "#EEF2F5";
const white = "#FFFFFF";
const darkText = "#20313D";
const methodColors = { A: "#376A9A", B: "#16807A", C: "#D39A2C", D: "#687784" };

function titleBand(sheet, range, text) {
  sheet.getRange(range).merge();
  const cell = sheet.getRange(range.split(":")[0]);
  cell.values = [[text]];
  cell.format = {
    fill: navy,
    font: { bold: true, color: white },
    verticalAlignment: "center",
  };
  sheet.getRange(range).format.rowHeight = 30;
}

function headerStyle(range) {
  range.format = {
    fill: teal,
    font: { bold: true, color: white },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#AAB8C2" },
  };
}

function sectionStyle(range) {
  range.format = {
    fill: lightBlue,
    font: { bold: true, color: navy },
    borders: { preset: "outside", style: "thin", color: "#AAB8C2" },
  };
}

for (const sheet of [readme, overall, compare, raw]) {
  sheet.showGridLines = false;
}

// README
titleBand(readme, "A1:H1", "Daphnet A–D逐被试主指标导出");
readme.getRange("A3:B10").values = [
  ["项目", "说明"],
  ["数据源", "daphnet_residual_calibration_ABCD_3seed_seed20260807"],
  ["被试", "S01、S02、S03、S05、S06、S07、S08、S09（排除S04、S10）"],
  ["主指标", "Accuracy、FoG Recall、Specificity、PR-AUC"],
  ["统计口径", "每个TCN种子先对3折取宏平均，再对3个种子计算均值±总体标准差"],
  ["阈值", "每个折/种子/方法均只在角色2/3验证集选择；不是逐被试阈值"],
  ["测试集", "角色0/1；全部模型和阈值冻结后统一测试"],
  ["生成日期", "2026-08-08"],
];
headerStyle(readme.getRange("A3:B3"));
readme.getRange("A4:A10").format = { fill: lightGray, font: { bold: true, color: darkText } };
readme.getRange("B4:B10").format.wrapText = true;
readme.getRange("A12:B16").values = [
  ["方法", "残差处理"],
  ...groups.map((group) => [group, methodDescriptions[group]]),
];
headerStyle(readme.getRange("A12:B12"));
readme.getRange("A13:A16").format = { fill: lightTeal, font: { bold: true, color: navy } };
readme.getRange("A1:A16").format.columnWidth = 18;
readme.getRange("B1:B16").format.columnWidth = 78;
readme.freezePanes.freezeRows(1);

// Overall sheet
titleBand(overall, "A1:I1", "A–D总体指标：3个种子的均值±标准差");
const overallMatrix = [["Metric"]];
for (const group of groups) overallMatrix[0].push(`${group} Mean`, `${group} SD`);
for (const [metricKey, metricLabel] of metricOrder) {
  const row = [metricLabel];
  for (const group of groups) {
    const source = overallRows.find(
      (item) => item.group === group && item.metric === metricKey,
    );
    row.push(Number(source.mean), Number(source.std));
  }
  overallMatrix.push(row);
}
overall.getRange("A3:I11").values = overallMatrix;
headerStyle(overall.getRange("A3:I3"));
overall.getRange("A4:A11").format = { fill: lightGray, font: { bold: true, color: darkText } };
overall.getRange("B4:I11").format.numberFormat = "0.00%";
overall.getRange("A3:I11").format.borders = {
  insideHorizontal: { style: "thin", color: "#D6DEE4" },
  bottom: { style: "thin", color: "#AAB8C2" },
};
overall.getRange("A1:A11").format.columnWidth = 23;
overall.getRange("B1:I11").format.columnWidth = 13;
overall.getRange("K2:O6").values = [
  ["Main Metric", "A", "B", "C", "D"],
  ["Accuracy", null, null, null, null],
  ["FoG Recall", null, null, null, null],
  ["Specificity", null, null, null, null],
  ["PR-AUC", null, null, null, null],
];
headerStyle(overall.getRange("K2:O2"));
const overviewMetricRows = { Accuracy: 4, "FoG Recall": 7, Specificity: 8, "PR-AUC": 10 };
const overviewMeanColumns = { A: "B", B: "D", C: "F", D: "H" };
for (let rowIndex = 3; rowIndex <= 6; rowIndex += 1) {
  const label = overall.getRange(`K${rowIndex}`).values[0][0];
  const sourceRow = overviewMetricRows[label];
  for (let groupIndex = 0; groupIndex < groups.length; groupIndex += 1) {
    const group = groups[groupIndex];
    const column = String.fromCharCode("L".charCodeAt(0) + groupIndex);
    overall.getRange(`${column}${rowIndex}`).formulas = [
      [`='Overall'!${overviewMeanColumns[group]}${sourceRow}`],
    ];
  }
}
overall.getRange("L3:O6").format.numberFormat = "0.00%";
const chart = overall.charts.add("bar", overall.getRange("K2:O6"));
chart.title = "总体主指标比较";
chart.hasLegend = true;
chart.xAxis = { axisType: "textAxis" };
chart.yAxis = { numberFormatCode: "0%", min: 0.3, max: 1.0 };
chart.setPosition("K8", "R24");
overall.freezePanes.freezeRows(3);

// Raw subject data
const rawHeaders = [
  "group",
  "subject_id",
  "accuracy_mean",
  "accuracy_std",
  "balanced_accuracy_mean",
  "balanced_accuracy_std",
  "precision_mean",
  "precision_std",
  "sensitivity_mean",
  "sensitivity_std",
  "specificity_mean",
  "specificity_std",
  "f1_mean",
  "f1_std",
  "auprc_mean",
  "auprc_std",
  "auroc_mean",
  "auroc_std",
];
const rawValues = [
  rawHeaders,
  ...subjectRows.map((row) =>
    rawHeaders.map((header) =>
      header === "group" || header === "subject_id" ? row[header] : Number(row[header]),
    ),
  ),
];
raw.getRange("A1:R33").values = rawValues;
headerStyle(raw.getRange("A1:R1"));
raw.getRange("C2:R33").format.numberFormat = "0.00%";
raw.getRange("A1:R33").format.borders = {
  insideHorizontal: { style: "thin", color: "#E0E6EA" },
  bottom: { style: "thin", color: "#AAB8C2" },
};
raw.getRange("A1:B33").format.columnWidth = 13;
raw.getRange("C1:R33").format.columnWidth = 18;
raw.getRange("A1:R1").format.rowHeight = 36;
raw.freezePanes.freezeRows(1);
raw.freezePanes.freezeColumns(2);
const rawTable = raw.tables.add("A1:R33", true, "SubjectMetricsData");
rawTable.style = "TableStyleMedium2";

// Subject comparison sheet: four compact 2x2 blocks.
titleBand(compare, "A1:M1", "逐被试主指标比较（单元格为3种子均值）");
const blocks = [
  { metric: primaryMetrics[0], startCol: "A", headerRow: 3, dataRow: 4 },
  { metric: primaryMetrics[2], startCol: "H", headerRow: 3, dataRow: 4 },
  { metric: primaryMetrics[1], startCol: "A", headerRow: 15, dataRow: 16 },
  { metric: primaryMetrics[3], startCol: "H", headerRow: 15, dataRow: 16 },
];
const columnNumber = (letter) => letter.charCodeAt(0) - "A".charCodeAt(0) + 1;
const columnLetter = (number) => String.fromCharCode("A".charCodeAt(0) + number - 1);

for (const block of blocks) {
  const start = columnNumber(block.startCol);
  const end = start + 5;
  const endCol = columnLetter(end);
  compare.getRange(`${block.startCol}${block.headerRow - 1}:${endCol}${block.headerRow - 1}`).merge();
  compare.getRange(`${block.startCol}${block.headerRow - 1}`).values = [[block.metric.label]];
  sectionStyle(compare.getRange(`${block.startCol}${block.headerRow - 1}:${endCol}${block.headerRow - 1}`));
  const headers = ["Subject", ...groups, "Best"];
  compare.getRange(`${block.startCol}${block.headerRow}:${endCol}${block.headerRow}`).values = [headers];
  headerStyle(compare.getRange(`${block.startCol}${block.headerRow}:${endCol}${block.headerRow}`));
  for (let subjectIndex = 0; subjectIndex < subjects.length; subjectIndex += 1) {
    const targetRow = block.dataRow + subjectIndex;
    compare.getRange(`${block.startCol}${targetRow}`).values = [[subjects[subjectIndex]]];
    for (let groupIndex = 0; groupIndex < groups.length; groupIndex += 1) {
      const targetCol = columnLetter(start + 1 + groupIndex);
      const group = groups[groupIndex];
      compare.getRange(`${targetCol}${targetRow}`).formulas = [[
        `=SUMIFS('Subject Data'!$${block.metric.meanColumn}$2:$${block.metric.meanColumn}$33,'Subject Data'!$A$2:$A$33,"${group}",'Subject Data'!$B$2:$B$33,${block.startCol}${targetRow})`,
      ]];
    }
    const firstMetricCol = columnLetter(start + 1);
    const lastMetricCol = columnLetter(start + 4);
    compare.getRange(`${endCol}${targetRow}`).formulas = [[
      `=INDEX(${firstMetricCol}$${block.headerRow}:${lastMetricCol}$${block.headerRow},1,MATCH(MAX(${firstMetricCol}${targetRow}:${lastMetricCol}${targetRow}),${firstMetricCol}${targetRow}:${lastMetricCol}${targetRow},0))`,
    ]];
  }
  const firstMetricCol = columnLetter(start + 1);
  const lastMetricCol = columnLetter(start + 4);
  compare.getRange(`${firstMetricCol}${block.dataRow}:${lastMetricCol}${block.dataRow + 7}`).format.numberFormat = "0.00%";
  compare.getRange(`${block.startCol}${block.dataRow}:${endCol}${block.dataRow + 7}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D6DEE4" },
    bottom: { style: "thin", color: "#AAB8C2" },
  };
  compare
    .getRange(`${firstMetricCol}${block.dataRow}:${lastMetricCol}${block.dataRow + 7}`)
    .conditionalFormats.add("colorScale", {
      colors: ["#F4CCCC", "#FFF2CC", "#D9EAD3"],
      thresholds: ["min", "50%", "max"],
    });
  compare.getRange(`${endCol}${block.dataRow}:${endCol}${block.dataRow + 7}`).format = {
    fill: lightTeal,
    font: { bold: true, color: navy },
  };
}
for (const column of ["A", "H"]) compare.getRange(`${column}1:${column}24`).format.columnWidth = 13;
for (const column of ["B", "C", "D", "E", "I", "J", "K", "L"]) {
  compare.getRange(`${column}1:${column}24`).format.columnWidth = 12;
}
for (const column of ["F", "M"]) compare.getRange(`${column}1:${column}24`).format.columnWidth = 12;
compare.getRange("G1:G24").format.columnWidth = 4;
compare.freezePanes.freezeRows(3);

// Markdown export with mean ± SD and best method per subject.
const percent = (mean, std) => `${(100 * Number(mean)).toFixed(2)} ± ${(100 * Number(std)).toFixed(2)}%`;
const md = [];
md.push("# Daphnet A–D逐被试主指标比较", "");
md.push(
  "统计口径：每个TCN种子先对3折取宏平均，再对3个种子计算均值±总体标准差。阈值来自每个折/种子的角色2/3验证集，不为单独被试重新选阈值。",
  "",
);
md.push("| 被试 | 方法 | Accuracy | FoG Recall | Specificity | PR-AUC |", "|---|---|---:|---:|---:|---:|");
for (const subject of subjects) {
  for (const group of groups) {
    const row = subjectRows.find((item) => item.subject_id === subject && item.group === group);
    md.push(
      `| ${subject} | ${group} | ${percent(row.accuracy_mean, row.accuracy_std)} | ${percent(row.sensitivity_mean, row.sensitivity_std)} | ${percent(row.specificity_mean, row.specificity_std)} | ${percent(row.auprc_mean, row.auprc_std)} |`,
    );
  }
}
md.push("", "## 每名被试的最佳方法（按均值）", "");
md.push("| 被试 | Accuracy | FoG Recall | Specificity | PR-AUC |", "|---|---|---|---|---|");
for (const subject of subjects) {
  const rows = subjectRows.filter((item) => item.subject_id === subject);
  const best = (key) => rows.reduce((left, right) => Number(right[key]) > Number(left[key]) ? right : left).group;
  md.push(`| ${subject} | ${best("accuracy_mean")} | ${best("sensitivity_mean")} | ${best("specificity_mean")} | ${best("auprc_mean")} |`);
}
md.push("", "方法定义：A=位置+尺度校准；B=位置+尺度校准后窗口中心化；C=仅尺度校准后窗口中心化；D=仅对原始重构误差窗口中心化。", "");

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
await fs.writeFile(outputMarkdown, md.join("\n"), "utf8");

const checks = [];
checks.push(
  await workbook.inspect({
    kind: "table",
    range: "Subject Compare!A1:M24",
    include: "values,formulas",
    tableMaxRows: 24,
    tableMaxCols: 13,
    maxChars: 8000,
  }),
);
checks.push(
  await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  }),
);
for (const check of checks) console.log(check.ndjson);

for (const [sheetName, range] of [
  ["README", "A1:H16"],
  ["Overall", "A1:R24"],
  ["Subject Compare", "A1:M24"],
  ["Subject Data", "A1:R33"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(
    path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputXlsx);
console.log(JSON.stringify({ outputXlsx, outputMarkdown, previewDir }, null, 2));
