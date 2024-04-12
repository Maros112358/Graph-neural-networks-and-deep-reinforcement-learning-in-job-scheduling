from bs4 import BeautifulSoup

# Read the HTML file
with open('Job Shop Instances and Solutions.html', 'r') as html_file, open('jssp_sizes.csv', 'w') as data_file:
    html_content = html_file.read()

    # Parse the HTML
    soup = BeautifulSoup(html_content, 'lxml')

    # Find all the <tr> tags
    rows = soup.find_all('tr')

    data_file.write("instance,jobs,machines\n")
    # Iterate over each row
    for row in rows:
        # Find all <td> tags within this row
        cells = row.find_all('td')
        # Extract the text from each <td>
        data = [cell.text for cell in cells]
        instance = data[0]
        jobs = data[1]
        machines = data[2]
        data_file.write(f"{instance},{jobs},{machines}\n")
