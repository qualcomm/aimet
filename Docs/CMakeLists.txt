install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DESTINATION .
      PATTERN "CMakeFiles*" EXCLUDE
      PATTERN "Makefile" EXCLUDE
      PATTERN "*.cmake" EXCLUDE
      )

# Laid out the way aimet-pages serves it: releases/<version>/ is the build as-is (every
# page noindexed, see conf.py) and releases/latest/ is a copy with that tag removed, so
# only latest gets indexed.
set(DOCS_RELEASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/releases/${SW_VERSION})
set(DOCS_LATEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/releases/latest)

add_custom_target(doc
      COMMAND ${CMAKE_COMMAND} -E env
      "${AIMET_PYTHONPATH}"
      "SW_VERSION=${SW_VERSION}"
      sphinx-build -v -T -b html ${CMAKE_BINARY_DIR}/Docs_SOURCE ${DOCS_RELEASE_DIR}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_latest_docs.py
      ${DOCS_RELEASE_DIR} ${DOCS_LATEST_DIR})

add_dependencies(doc
    copy_doc_source copy_examples)

add_custom_target(copy_doc_source
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/Docs_SOURCE)
