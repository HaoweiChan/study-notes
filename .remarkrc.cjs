module.exports = {
  plugins: [
    'remark-frontmatter',
    ['remark-validate-frontmatter', {properties: {title: {type: 'string'}, date: {type: 'string'}}}],
    'remark-lint',
    'remark-preset-lint-recommended'
  ]
}
