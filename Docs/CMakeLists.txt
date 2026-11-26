install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DESTINATION .
      PATTERN "CMakeFiles*" EXCLUDE
      PATTERN "Makefile" EXCLUDE
      PATTERN "*.cmake" EXCLUDE
      )

# Laid out the way aimet-pages serves it, so only latest gets indexed (see
# make_publish_layout.py): releases/<version>/ is noindexed with no canonical links or
# sitemap.xml, and releases/latest/ is indexable with self-referential canonical links and
# sitemap.xml. Neither has a robots.txt.
set(DOCS_RELEASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/releases/${SW_VERSION})
set(DOCS_LATEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/releases/latest)

add_custom_target(doc
      COMMAND ${CMAKE_COMMAND} -E env
      "${AIMET_PYTHONPATH}"
      "SW_VERSION=${SW_VERSION}"
      sphinx-build -v -T -b html ${CMAKE_BINARY_DIR}/Docs_SOURCE ${DOCS_RELEASE_DIR}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_publish_layout.py
      ${DOCS_RELEASE_DIR} ${DOCS_LATEST_DIR})

add_dependencies(doc
    copy_doc_source copy_examples)

add_custom_target(copy_doc_source
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/Docs_SOURCE)
