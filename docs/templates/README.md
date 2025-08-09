# Documentation Templates

This directory contains templates for creating consistent, high-quality documentation for pysearch.

## Available Templates

### 📚 Content Templates

- **[User Guide Template](user-guide-template.md)** - For feature guides and how-to documentation
- **[API Reference Template](api-reference-template.md)** - For detailed API documentation
- **[Tutorial Template](tutorial-template.md)** - For step-by-step learning materials

### 📋 Process Templates

- **[Documentation Review Checklist](#documentation-review-checklist)** - Quality assurance checklist
- **[Tutorial Review Checklist](#tutorial-review-checklist)** - Tutorial-specific review items

---

## How to Use Templates

### 1. Choose the Right Template

| Document Type | Template | When to Use |
|---------------|----------|-------------|
| **Feature Guide** | [User Guide Template](user-guide-template.md) | Explaining how to use a feature |
| **API Documentation** | [API Reference Template](api-reference-template.md) | Documenting classes, functions, modules |
| **Learning Material** | [Tutorial Template](tutorial-template.md) | Step-by-step instructions |

### 2. Copy and Customize

```bash
# Copy template to your new document
cp docs/templates/user-guide-template.md docs/my-new-guide.md

# Edit the template
# Replace all [placeholders] with actual content
# Follow the structure but adapt as needed
```

### 3. Follow the Style Guide

All templates follow the [Documentation Style Guide](../style-guide.md). Key principles:

- **Clear structure** with consistent headings
- **Complete examples** that actually work
- **Progressive complexity** from simple to advanced
- **Helpful cross-references** to related content

---

## Template Guidelines

### Placeholder Convention

Templates use this placeholder format:

- `[Feature Name]` - Replace with actual feature name
- `[Description]` - Replace with actual description
- `[Code Example]` - Replace with working code
- `[Version]` - Replace with version number
- `[Date]` - Replace with current date

### Required Sections

All templates include these essential sections:

1. **Title and Description** - What this document covers
2. **Table of Contents** - Navigation for longer documents
3. **Prerequisites** - What readers need to know/have
4. **Examples** - Working, tested code examples
5. **See Also** - Links to related documentation

### Optional Sections

Include these sections when relevant:

- **Quick Reference** - Summary tables or lists
- **Troubleshooting** - Common issues and solutions
- **Performance** - Performance considerations
- **Migration** - Upgrade/migration information

---

## Writing Process

### 1. Planning Phase

Before writing:

- [ ] **Identify audience** - Who will read this?
- [ ] **Define scope** - What will be covered?
- [ ] **Choose template** - Which template fits best?
- [ ] **Gather examples** - Collect working code examples
- [ ] **Review related docs** - Check for overlaps or gaps

### 2. Writing Phase

While writing:

- [ ] **Follow template structure** - Use provided sections
- [ ] **Replace all placeholders** - No [brackets] in final version
- [ ] **Test all examples** - Ensure code actually works
- [ ] **Add cross-references** - Link to related content
- [ ] **Include error handling** - Show how to handle failures

### 3. Review Phase

Before publishing:

- [ ] **Self-review** - Read through completely
- [ ] **Check examples** - Test all code snippets
- [ ] **Verify links** - Ensure all links work
- [ ] **Spell check** - Use automated tools
- [ ] **Peer review** - Get feedback from others

---

## Quality Standards

### Content Quality

- **Accuracy**: All information must be correct and current
- **Completeness**: Cover the topic thoroughly
- **Clarity**: Use simple, clear language
- **Examples**: Include practical, working examples

### Technical Quality

- **Code Examples**: Must be complete and runnable
- **Error Handling**: Show how to handle common errors
- **Performance**: Include performance considerations
- **Best Practices**: Demonstrate recommended approaches

### Formatting Quality

- **Consistent Structure**: Follow template organization
- **Proper Markdown**: Use correct markdown syntax
- **Code Formatting**: Use appropriate language tags
- **Link Format**: Use descriptive link text

---

## Review Checklists

### Documentation Review Checklist

Use this checklist for all documentation:

#### Structure and Organization

- [ ] **Title** is clear and descriptive
- [ ] **Table of contents** is present (for docs >3 sections)
- [ ] **Headings** follow consistent hierarchy (H1 → H2 → H3)
- [ ] **Sections** are logically organized
- [ ] **Cross-references** link to related content

#### Content Quality

- [ ] **Audience** is clearly defined
- [ ] **Prerequisites** are listed
- [ ] **Examples** are complete and tested
- [ ] **Error handling** is covered
- [ ] **Best practices** are included

#### Technical Accuracy

- [ ] **Code examples** run without errors
- [ ] **API references** match current implementation
- [ ] **Version information** is current
- [ ] **Links** all work correctly
- [ ] **Commands** produce expected output

#### Writing Quality

- [ ] **Grammar** and spelling are correct
- [ ] **Tone** is consistent and appropriate
- [ ] **Terminology** is used consistently
- [ ] **Active voice** is used where possible
- [ ] **Jargon** is explained when first used

#### Formatting

- [ ] **Markdown syntax** is correct
- [ ] **Code blocks** have language specification
- [ ] **Lists** use consistent formatting
- [ ] **Tables** are properly formatted
- [ ] **Images** have alt text (if applicable)

### Tutorial Review Checklist

Additional items for tutorials:

#### Learning Design

- [ ] **Learning objectives** are clearly stated
- [ ] **Prerequisites** are appropriate for audience
- [ ] **Difficulty progression** is gradual
- [ ] **Exercises** reinforce learning
- [ ] **Solutions** are provided for exercises

#### Instructional Quality

- [ ] **Steps** are clear and numbered
- [ ] **Expected output** is shown
- [ ] **Common mistakes** are addressed
- [ ] **Troubleshooting** section is included
- [ ] **Next steps** are suggested

#### Practical Application

- [ ] **Real-world examples** are used
- [ ] **Complete project** demonstrates concepts
- [ ] **Variations** are suggested
- [ ] **Extensions** are provided for advanced users

---

## Template Maintenance

### Updating Templates

Templates should be updated when:

- **API changes** affect documented interfaces
- **New features** require additional sections
- **User feedback** suggests improvements
- **Style guide** changes require updates

### Version Control

- **Track changes** to templates with meaningful commit messages
- **Review updates** before merging to main branch
- **Notify contributors** of significant template changes
- **Update documentation** that uses old template versions

### Community Feedback

We welcome feedback on templates:

- **Suggest improvements** via GitHub issues
- **Share examples** of good documentation
- **Report problems** with template usage
- **Contribute new templates** for missing use cases

---

## Examples of Good Documentation

### Internal Examples

These pysearch docs follow our templates well:

- **[Usage Guide](../usage.md)** - Comprehensive feature guide
- **[API Reference](../api-reference.md)** - Complete API documentation
- **[Tutorial 01](../tutorials/01_getting_started.py)** - Well-structured tutorial

### External Examples

Great documentation from other projects:

- **[Requests Documentation](https://docs.python-requests.org/)** - Clear, practical examples
- **[FastAPI Docs](https://fastapi.tiangolo.com/)** - Excellent tutorial progression
- **[Django Documentation](https://docs.djangoproject.com/)** - Comprehensive reference

---

## Contributing Templates

### Creating New Templates

To create a new template:

1. **Identify the need** - What type of documentation is missing?
2. **Study existing templates** - Follow established patterns
3. **Create draft template** - Include all necessary sections
4. **Test with real content** - Use template to create actual documentation
5. **Get feedback** - Review with team members
6. **Submit pull request** - Include rationale and examples

### Template Requirements

New templates must:

- **Follow style guide** - Consistent with existing templates
- **Include placeholders** - Clear [placeholder] format
- **Provide examples** - Show how to use each section
- **Be complete** - Cover all necessary aspects
- **Be tested** - Used to create real documentation

### Review Process

Template changes go through:

1. **Initial review** - Check completeness and consistency
2. **Testing** - Use template to create sample documentation
3. **Community feedback** - Get input from documentation team
4. **Final approval** - Merge after addressing feedback

---

## Support and Questions

### Getting Help

If you need help with templates:

1. **Check existing examples** - Look at current documentation
2. **Review style guide** - Follow established standards
3. **Ask in discussions** - Get community input
4. **Create an issue** - Report problems or suggest improvements

### Contact Information

- **Documentation Team**: @docs-team
- **GitHub Discussions**: [Documentation Category](https://github.com/org/pysearch/discussions/categories/documentation)
- **Issues**: [Documentation Issues](https://github.com/org/pysearch/issues?q=label%3Adocumentation)

---

## Changelog

### Recent Updates

- **2024-01-15**: Added tutorial template with exercise sections
- **2024-01-10**: Updated API reference template with type hints
- **2024-01-05**: Created initial template collection
- **2024-01-01**: Established template directory structure

### Planned Improvements

- **FAQ template** - For frequently asked questions
- **Troubleshooting template** - For problem-solving guides
- **Migration guide template** - For version upgrade documentation
- **Integration template** - For third-party integration guides

---

**Happy documenting!** 📝

*These templates are living documents. Please suggest improvements and share your experiences using them.*
