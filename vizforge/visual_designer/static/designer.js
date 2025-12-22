// VizForge Visual Designer JavaScript

// Global state
let currentData = null;
let currentConfig = {
    chart_type: null,
    title: '',
    properties: {},
    data_source: null,
    filters: []
};
let dataColumns = [];
let numericColumns = [];
let categoricalColumns = [];

// Initialize on page load
$(document).ready(function() {
    loadChartTypes();
});

// Load available chart types from API
function loadChartTypes() {
    $.ajax({
        url: '/api/chart_types',
        method: 'GET',
        success: function(categories) {
            renderChartTypes(categories);
        },
        error: function(xhr) {
            showAlert('Error loading chart types', 'danger');
        }
    });
}

// Render chart types by category
function renderChartTypes(categories) {
    const container = $('#chart-types-container');
    container.empty();

    for (const [category, types] of Object.entries(categories)) {
        const categoryDiv = $('<div>').addClass('chart-category');

        // Category title
        const title = $('<div>')
            .addClass('chart-category-title')
            .html(`<i class="fas fa-folder-open"></i> ${category}`);
        categoryDiv.append(title);

        // Chart type cards
        types.forEach(type => {
            const card = $('<div>')
                .addClass('chart-type-card')
                .attr('data-chart-type', type.value)
                .html(`<i class="fas fa-chart-bar"></i> ${type.label}`)
                .click(function() {
                    selectChartType(type.value);
                });
            categoryDiv.append(card);
        });

        container.append(categoryDiv);
    }
}

// Upload data file
function uploadData(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    showLoadingOverlay('Uploading data...');

    $.ajax({
        url: '/api/upload_data',
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            hideLoadingOverlay();
            currentData = response;
            dataColumns = response.columns;
            updateDataInfo(response);
            showAlert(`Data uploaded: ${response.rows} rows, ${response.columns.length} columns`, 'success');

            // Get column types
            getDataInfo();
        },
        error: function(xhr) {
            hideLoadingOverlay();
            const error = xhr.responseJSON?.error || 'Upload failed';
            showAlert(error, 'danger');
        }
    });
}

// Get data information
function getDataInfo() {
    $.ajax({
        url: '/api/data_info',
        method: 'GET',
        success: function(response) {
            numericColumns = response.numeric_columns;
            categoricalColumns = response.categorical_columns;
        }
    });
}

// Update data info display
function updateDataInfo(data) {
    const infoDiv = $('#data-info');
    $('#data-filename').html(`<strong>${data.filename}</strong>`);
    $('#data-rows').html(`${data.rows} rows × ${data.columns.length} columns`);
    infoDiv.fadeIn();

    currentConfig.data_source = data.filename;
}

// Select chart type
function selectChartType(chartType) {
    if (!currentData) {
        showAlert('Please upload data first', 'warning');
        return;
    }

    currentConfig.chart_type = chartType;

    // Get properties for this chart type
    $.ajax({
        url: '/api/chart_properties',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ chart_type: chartType }),
        success: function(response) {
            renderProperties(response.properties);
            $('.empty-state').hide();
        },
        error: function(xhr) {
            showAlert('Error loading chart properties', 'danger');
        }
    });
}

// Render property editor
function renderProperties(properties) {
    const container = $('#properties-container');
    container.empty();

    // Group properties by section
    const commonProps = properties.filter(p => ['title', 'width', 'height', 'theme'].includes(p.name));
    const dataProps = properties.filter(p => !['title', 'width', 'height', 'theme'].includes(p.name));

    // Common properties section
    if (commonProps.length > 0) {
        const section = createPropertySection('Common', commonProps);
        container.append(section);
    }

    // Data properties section
    if (dataProps.length > 0) {
        const section = createPropertySection('Data', dataProps);
        container.append(section);
    }

    // Preview button
    const previewBtn = $('<button>')
        .addClass('btn btn-primary btn-preview')
        .html('<i class="fas fa-eye"></i> Preview Chart')
        .click(previewChart);
    container.append(previewBtn);
}

// Create property section
function createPropertySection(title, properties) {
    const section = $('<div>').addClass('property-section');
    const sectionTitle = $('<div>').addClass('property-section-title').text(title);
    section.append(sectionTitle);

    properties.forEach(prop => {
        const group = createPropertyInput(prop);
        section.append(group);
    });

    return section;
}

// Create property input based on type
function createPropertyInput(prop) {
    const group = $('<div>').addClass('property-group');

    const label = $('<label>')
        .addClass('property-label')
        .text(prop.label);

    if (prop.required) {
        label.append('<span class="required-asterisk">*</span>');
    }

    group.append(label);

    let input;

    if (prop.type === 'string') {
        input = $('<input>')
            .attr('type', 'text')
            .addClass('property-input form-control')
            .val(prop.default || '')
            .on('change', function() {
                currentConfig.properties[prop.name] = $(this).val();
            });
    }

    else if (prop.type === 'number') {
        input = $('<input>')
            .attr('type', 'number')
            .addClass('property-input form-control')
            .val(prop.default || 0);

        if (prop.min_value !== null) {
            input.attr('min', prop.min_value);
        }
        if (prop.max_value !== null) {
            input.attr('max', prop.max_value);
        }

        input.on('change', function() {
            currentConfig.properties[prop.name] = parseFloat($(this).val());
        });
    }

    else if (prop.type === 'boolean') {
        input = $('<div>').addClass('form-check');
        const checkbox = $('<input>')
            .attr('type', 'checkbox')
            .addClass('form-check-input')
            .attr('id', `prop-${prop.name}`)
            .prop('checked', prop.default || false)
            .on('change', function() {
                currentConfig.properties[prop.name] = $(this).is(':checked');
            });
        const checkLabel = $('<label>')
            .addClass('form-check-label')
            .attr('for', `prop-${prop.name}`)
            .text('Enable');

        input.append(checkbox, checkLabel);
    }

    else if (prop.type === 'select') {
        input = $('<select>')
            .addClass('property-input form-select');

        if (prop.options) {
            prop.options.forEach(opt => {
                const option = $('<option>')
                    .val(opt)
                    .text(opt);
                if (opt === prop.default) {
                    option.attr('selected', true);
                }
                input.append(option);
            });
        }

        input.on('change', function() {
            currentConfig.properties[prop.name] = $(this).val();
        });
    }

    else if (prop.type === 'column') {
        input = $('<select>')
            .addClass('property-input form-select');

        input.append('<option value="">-- Select Column --</option>');

        dataColumns.forEach(col => {
            const option = $('<option>').val(col).text(col);
            input.append(option);
        });

        input.on('change', function() {
            const value = $(this).val();
            currentConfig.properties[prop.name] = value || null;
        });
    }

    else if (prop.type === 'columns') {
        // Multi-select for columns
        input = $('<div>').addClass('column-checkbox-group');

        dataColumns.forEach(col => {
            const item = $('<div>').addClass('column-checkbox-item form-check');
            const checkbox = $('<input>')
                .attr('type', 'checkbox')
                .addClass('form-check-input')
                .attr('id', `col-${col}`)
                .val(col)
                .on('change', function() {
                    updateMultiSelectColumns(prop.name);
                });
            const checkLabel = $('<label>')
                .addClass('form-check-label')
                .attr('for', `col-${col}`)
                .text(col);

            item.append(checkbox, checkLabel);
            input.append(item);
        });
    }

    else {
        input = $('<input>')
            .attr('type', 'text')
            .addClass('property-input form-control')
            .val(prop.default || '');
    }

    group.append(input);

    // Add description if available
    if (prop.description) {
        const desc = $('<div>')
            .addClass('property-description')
            .text(prop.description);
        group.append(desc);
    }

    // Set initial value
    if (prop.default !== null && prop.default !== undefined) {
        currentConfig.properties[prop.name] = prop.default;
    }

    return group;
}

// Update multi-select columns
function updateMultiSelectColumns(propName) {
    const selected = [];
    $('.column-checkbox-group input:checked').each(function() {
        selected.push($(this).val());
    });
    currentConfig.properties[propName] = selected;
}

// Preview chart
function previewChart() {
    if (!currentConfig.chart_type) {
        showAlert('Please select a chart type', 'warning');
        return;
    }

    // Validate required properties
    // (Add validation logic here if needed)

    showLoadingOverlay('Generating preview...');

    $.ajax({
        url: '/api/preview_chart',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(currentConfig),
        success: function(response) {
            hideLoadingOverlay();
            displayChart(response.html);
        },
        error: function(xhr) {
            hideLoadingOverlay();
            const error = xhr.responseJSON?.error || 'Preview failed';
            showAlert(error, 'danger');
        }
    });
}

// Display chart in canvas
function displayChart(html) {
    $('.empty-state').hide();
    $('#chart-preview').html(html).show().addClass('fade-in');
}

// Export Python code
function exportCode() {
    if (!currentConfig.chart_type) {
        showAlert('No chart to export', 'warning');
        return;
    }

    const includeImports = $('#includeImports').is(':checked');
    const includeDataLoading = $('#includeDataLoading').is(':checked');

    $.ajax({
        url: '/api/generate_code',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            ...currentConfig,
            include_imports: includeImports,
            include_data_loading: includeDataLoading
        }),
        success: function(response) {
            $('#generated-code code').text(response.code);
            const modal = new bootstrap.Modal($('#codeModal'));
            modal.show();
        },
        error: function(xhr) {
            const error = xhr.responseJSON?.error || 'Code generation failed';
            showAlert(error, 'danger');
        }
    });
}

// Copy code to clipboard
function copyCode() {
    const code = $('#generated-code code').text();
    navigator.clipboard.writeText(code).then(function() {
        showAlert('Code copied to clipboard!', 'success');
    });
}

// Export image
function exportImage() {
    if (!currentConfig.chart_type) {
        showAlert('No chart to export', 'warning');
        return;
    }

    showLoadingOverlay('Exporting image...');

    $.ajax({
        url: '/api/export_chart',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            ...currentConfig,
            format: 'png'
        }),
        success: function(response) {
            hideLoadingOverlay();

            // Create download link
            const link = document.createElement('a');
            link.href = response.image;
            link.download = `${currentConfig.chart_type}_chart.png`;
            link.click();

            showAlert('Image exported successfully!', 'success');
        },
        error: function(xhr) {
            hideLoadingOverlay();
            const error = xhr.responseJSON?.error || 'Export failed';
            showAlert(error, 'danger');
        }
    });
}

// Show help modal
function showHelp() {
    const modal = new bootstrap.Modal($('#helpModal'));
    modal.show();
}

// Show loading overlay
function showLoadingOverlay(message = 'Loading...') {
    const overlay = $('<div>')
        .addClass('loading-overlay')
        .html(`
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="mt-2">${message}</div>
            </div>
        `);

    $('#canvas-area').append(overlay);
}

// Hide loading overlay
function hideLoadingOverlay() {
    $('.loading-overlay').remove();
}

// Show alert message
function showAlert(message, type = 'info') {
    const alert = $('<div>')
        .addClass(`alert alert-${type} alert-dismissible fade show alert-floating`)
        .attr('role', 'alert')
        .html(`
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `);

    $('body').append(alert);

    // Auto-dismiss after 5 seconds
    setTimeout(function() {
        alert.alert('close');
    }, 5000);
}
