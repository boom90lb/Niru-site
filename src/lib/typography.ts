/**
 * Typographic quotes for text that does not pass through Markdown, such as
 * frontmatter and data files: an apostrophe inside or at the end of a word
 * becomes a right single quotation mark.
 */
export function smart(text: string): string {
  return text.replace(/(\w)'(\w|\s|$)/g, "$1’$2").replace(/(\s|^)'(\w)/g, "$1‘$2");
}

/** Split text so that each hyphenated word can be kept on one line. */
export function hyphenated(text: string): Array<{ text: string; keep: boolean }> {
  return text
    .split(/(\S*\w-\w\S*)/)
    .map((part, i) => ({ text: part, keep: i % 2 === 1 }))
    .filter((part) => part.text !== "");
}
